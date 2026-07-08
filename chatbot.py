import os
import re
import json
import time
import threading
import base64
import tempfile
import urllib.parse
import requests
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from gtts import gTTS
import asyncio
import edge_tts
from pypdf import PdfReader

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# ─────────────────────────────────────────────
# NLLB-200 (Hugging Face) — traduction dédiée pour le wolof
# ─────────────────────────────────────────────
# Llama 3.3 a une connaissance très limitée du wolof. NLLB-200 (Meta) est
# lui entraîné spécifiquement pour la traduction et supporte officiellement
# le wolof (code "wol_Latn") : bien meilleure qualité que de demander à un
# LLM généraliste de traduire. On l'utilise en priorité, avec repli sur la
# traduction via Llama+glossaire si la clé HF est absente ou l'appel échoue.
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
# Modèle par défaut : version de NLLB-200 fine-tunée spécifiquement sur du
# wolof/français/anglais (bilalfaye/english-wolof-french-translation),
# donc plus précise sur le wolof que le NLLB-200 générique. Reste
# surchargeable via la variable d'env NLLB_MODEL si besoin (ex. repli sur
# "facebook/nllb-200-distilled-600M" ou passage à la version 1.3B).
NLLB_MODEL = os.getenv("NLLB_MODEL", "bilalfaye/nllb-200-distilled-600M-wo-fr-en")
NLLB_API_URL = f"https://api-inference.huggingface.co/models/{NLLB_MODEL}"
# Codes de langue NLLB standards ; certains modèles fine-tunés utilisent
# parfois des codes custom dans leur tokenizer — surchargeables sans
# retoucher le code si jamais l'API renvoie une erreur de langue.
NLLB_LANG_CODES = {
    "fr": os.getenv("NLLB_FR_CODE", "fra_Latn"),
    "wl": os.getenv("NLLB_WL_CODE", "wol_Latn"),
}

if not HF_API_KEY:
    print(
        "[WARN] HF_API_KEY absente : la traduction wolof utilisera le "
        "repli Llama+glossaire (moins fiable que NLLB-200). Définis "
        "HF_API_KEY (clé Hugging Face gratuite) pour activer NLLB."
    )

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY manquant")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────
# IMAGE CONFIG
# ─────────────────────────────────────────────
TYPE_PROMPTS = {
    "logo": "minimalist professional flat vector logo...",
    "icon": "simple modern app icon...",
    "illustration": "vibrant detailed african art illustration...",
    "photo": "photorealistic DSLR photo...",
    "pattern": "seamless african kente textile pattern...",
    "banner": "modern marketing banner...",
    "avatar": "professional portrait photo...",
    "poster": "eye-catching poster design...",
    "general": "high quality digital art..."
}

TYPE_SIZES = {
    "logo": (1024, 1024),
    "icon": (512, 512),
    "illustration": (768, 1024),
    "photo": (1024, 1024),
    "pattern": (1024, 1024),
    "banner": (1024, 512),
    "avatar": (512, 512),
    "poster": (768, 1024),
    "general": (1024, 1024),
}

# ─────────────────────────────────────────────
# IMAGE INTENT DETECTION
# ─────────────────────────────────────────────
def detect_image_intent(msg: str):
    msg = msg.lower()
    triggers = ["logo", "image", "dessine", "crée", "avatar", "poster"]
    if not any(t in msg for t in triggers):
        return None
    return {
        "type": "general",
        "visual_prompt": msg,
        "confirmation_message": "🎨 Image générée !"
    }

# ─────────────────────────────────────────────
# LANGUAGE / SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es Yelen AI, un assistant qui parle français et wolof.\n"
    "Détecte automatiquement la langue du message de l'utilisateur "
    "(français ou wolof) et réponds TOUJOURS dans cette même langue.\n"
    "Si l'utilisateur écrit en wolof, fais de ton mieux pour répondre "
    "entièrement en wolof, même si ta maîtrise du wolof est imparfaite : "
    "ne bascule pas en français sauf si l'utilisateur te le demande "
    "explicitement ou s'il mélange lui-même les deux langues.\n"
    "Si l'utilisateur écrit en français, réponds en français."
)

# ─────────────────────────────────────────────
# WOLOF : GLOSSAIRE + TRADUCTION "SANDWICH"
# ─────────────────────────────────────────────
# Llama 3.3 a une connaissance très limitée du wolof (corpus d'entraînement
# quasi inexistant pour cette langue), donc :
#   1) il comprend mal certains mots/expressions wolof envoyés par
#      l'utilisateur (contresens, incompréhension totale)
#   2) il génère un wolof approximatif, mélangé de français, avec un
#      vocabulaire parfois inventé
#
# Stratégie retenue : "traduction sandwich"
#   wolof (utilisateur) --[traduction guidée par glossaire]--> français
#   --[LLM répond normalement en français, tâche qu'il maîtrise bien]-->
#   français --[traduction guidée par glossaire]--> wolof (réponse finale)
#
# Une tâche de TRADUCTION (contrainte, avec du contexte lexical fourni)
# est beaucoup plus fiable pour un LLM qu'une GÉNÉRATION libre dans une
# langue peu représentée dans son corpus. On combine ça avec un glossaire
# bilingue injecté dans le prompt pour forcer un vocabulaire correct et
# cohérent des deux côtés (compréhension ET génération).

WOLOF_FR_GLOSSARY = """
Glossaire wolof ↔ français (vocabulaire courant, à utiliser en priorité
pour la traduction — respecte cette terminologie plutôt que d'improviser) :

Salutations / politesse :
- Nanga def / Na nga def = Comment vas-tu / Comment ça va
- Maa ngi fi (rekk) = Je vais bien (ça va)
- Jërejëf = Merci
- Amul solo = De rien / Ce n'est rien
- Baal ma = Excuse-moi / Pardon
- Ba beneen = À bientôt / Au revoir

Oui / non / affirmations :
- Waaw = Oui
- Déedéet = Non
- Dara = Rien
- Baax na = D'accord / C'est bien
- Xam naa = Je sais
- Xamuma = Je ne sais pas
- Mën naa = Je peux
- Mënuma = Je ne peux pas

Mots interrogatifs :
- Naka = Comment
- Lan / Lu = Quoi
- Ndax = Est-ce que
- Lu tax = Pourquoi
- Fan / Ana = Où
- Kañ = Quand
- Kan = Qui
- Ñaata = Combien

Pronoms / personnes :
- Man = Moi
- Yow = Toi
- Moom = Lui / Elle
- Nun = Nous
- Yeen = Vous
- Ñoom = Eux

Verbes / expressions utiles pour un assistant :
- Dama bëgg / Da nga bëgg = Je veux / Tu veux
- Dinaa (+ verbe) = Je vais (futur proche)
- Wax = Parler / Dire
- Dégg = Comprendre / Entendre
- Xool = Regarder
- Bind = Écrire
- Jàng = Lire / Étudier
- Yëg = Ressentir / Faire savoir
- Bul jaaxle = Ne t'inquiète pas
- Léegi = Maintenant
- Tey = Aujourd'hui
- Ëllëg = Demain
- Démb = Hier
- Su fekkee = Si (condition)
- Bu bëgge = Si tu veux
""".strip()


# ─────────────────────────────────────────────
# GLOSSAIRE APPRIS EN CONVERSATION (mémoire persistante)
# ─────────────────────────────────────────────
# Quand l'utilisateur enseigne un mot ("X veut dire Y"), on le sauvegarde
# ici pour que le bot s'en souvienne dans les conversations futures — pas
# seulement dans l'historique de la conversation en cours (le modèle n'a
# aucune mémoire entre les requêtes HTTP).
#
# NB : c'est un glossaire GLOBAL (partagé entre tous les utilisateurs), il
# n'y a pas de notion d'utilisateur/session dans cette API pour l'instant.
# NB2 : sur un hébergeur avec disque éphémère (ex. Render free tier), ce
# fichier peut être réinitialisé à chaque redéploiement — pour une mémoire
# vraiment durable il faudrait une vraie base de données.
LEARNED_GLOSSARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learned_glossary.json"
)
_glossary_lock = threading.Lock()


def _load_learned_glossary() -> dict:
    try:
        with open(LEARNED_GLOSSARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_learned_glossary(data: dict):
    tmp_path = LEARNED_GLOSSARY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LEARNED_GLOSSARY_PATH)


def add_learned_words(pairs: list) -> list:
    """
    Ajoute une liste de {"wolof": ..., "french": ...} au glossaire appris.
    Retourne la liste des paires effectivement ajoutées/mises à jour.
    """
    added = []
    with _glossary_lock:
        current = _load_learned_glossary()
        for pair in pairs:
            wl = (pair.get("wolof") or "").strip()
            fr = (pair.get("french") or "").strip()
            if not wl or not fr:
                continue
            current[wl.lower()] = {"wolof": wl, "french": fr}
            added.append({"wolof": wl, "french": fr})
        if added:
            _save_learned_glossary(current)
    return added


def get_full_glossary_text() -> str:
    """Glossaire statique + mots appris en conversation, prêt pour un prompt."""
    learned = _load_learned_glossary()
    if not learned:
        return WOLOF_FR_GLOSSARY
    learned_lines = "\n".join(
        f"- {v['wolof']} = {v['french']}" for v in learned.values()
    )
    return (
        f"{WOLOF_FR_GLOSSARY}\n\n"
        "Mots appris récemment auprès de l'utilisateur (prioritaires, "
        "à utiliser tels quels) :\n"
        f"{learned_lines}"
    )


def get_learned_words() -> set:
    return set(_load_learned_glossary().keys())


# ─────────────────────────────────────────────
# DÉTECTION D'UN MESSAGE "ENSEIGNEMENT" (l'utilisateur apprend un mot au bot)
# ─────────────────────────────────────────────
TEACHING_MARKERS = [
    "veut dire", "ça veut dire", "ca veut dire", "veux dire",
    "signifie", "sinifie", "ce qui veut dire", "mooy", " = ",
    "retiens que", "retiens ça", "apprends", "en wolof on dit",
    "en wolof ça dit", "ça se dit",
]


def looks_like_teaching(text: str) -> bool:
    lowered = (text or "").lower()
    return any(m in lowered for m in TEACHING_MARKERS)


def extract_taught_pairs(text: str):
    """
    Utilise le LLM pour extraire les paires wolof/français que l'utilisateur
    est en train d'enseigner dans son message (ex: "Jamm veut dire paix").
    Retourne une liste de {"wolof":..., "french":...} (peut être vide).
    Appelé uniquement quand looks_like_teaching() a déjà pré-filtré, pour
    limiter les appels API inutiles.
    """
    instruction = (
        "L'utilisateur est en train d'apprendre du vocabulaire wolof à un "
        "assistant. Extrais toutes les paires mot/expression-wolof et "
        "leur traduction française présentes dans son message.\n"
        "Réponds STRICTEMENT avec un tableau JSON, sans aucun texte "
        "autour, sans balises markdown, au format exact :\n"
        '[{"wolof": "...", "french": "..."}]\n'
        "Si aucune paire d'enseignement claire n'est présente, réponds : []"
    )
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw = (r.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [p for p in parsed if isinstance(p, dict)]
        return []
    except Exception as e:
        print("[TEACHING EXTRACT ERROR]", e)
        return []


def detect_wolof(text: str) -> bool:
    """
    Heuristique locale (rapide, sans appel API) pour repérer un message
    probablement écrit en wolof, en cherchant des marqueurs très fréquents
    (mots-outils, pronoms, interrogatifs) plutôt qu'un mot-clé isolé — plus
    fiable que l'ancienne approche par mots-clés ponctuels.
    Ce n'est qu'un pré-filtre : en cas de doute on part du principe que
    c'est du français, et le SYSTEM_PROMPT sert de filet de sécurité pour
    les cas mixtes ou mal détectés.
    """
    if not text:
        return False

    markers = [
        "nanga", "def", "jërejëf", "jerejef", "waaw", "déedéet", "deedeet",
        "dara", "naka", "ndax", "lu tax", "fan", "kañ", "kan", "ñaata",
        "naata", "man", "yow", "moom", "nun", "yeen", "ñoom", "dama",
        "bëgg", "begg", "dinaa", "wax", "dégg", "degg", "xool", "bind",
        "jàng", "jang", "yëg", "yeg", "bul jaaxle", "léegi", "leegi",
        "tey", "ëllëg", "ellëg", "ellek", "démb", "demb", "baal ma",
        "mangi", "maa ngi", "xam naa", "xamuma", "mën naa", "menna",
        "baax na",
    ]

    # Les mots appris en conversation comptent aussi comme marqueurs —
    # sinon un mot enseigné par l'utilisateur ne serait jamais reconnu
    # comme wolof dans les messages suivants.
    markers = markers + list(get_learned_words())

    lowered = text.lower()
    hits = 0
    for m in markers:
        if not m:
            continue
        if re.search(rf"\b{re.escape(m)}\b", lowered):
            hits += 1
            if hits >= 1:
                return True
    return False


def _nllb_translate(text: str, src: str, tgt: str, max_retries: int = 2):
    """
    Appelle NLLB-200 via l'API d'inférence Hugging Face.
    src/tgt : "fr" ou "wl". Retourne le texte traduit, ou None si
    indisponible (pas de clé, erreur, modèle en cours de chargement après
    plusieurs tentatives) — auquel cas l'appelant doit basculer sur le
    repli LLM+glossaire.
    """
    if not HF_API_KEY:
        return None

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": NLLB_LANG_CODES[src],
            "tgt_lang": NLLB_LANG_CODES[tgt],
        },
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(NLLB_API_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data and "translation_text" in data[0]:
                    result = data[0]["translation_text"].strip()
                    return result if result else None
                print("[NLLB WARN] réponse inattendue :", data)
                return None

            if resp.status_code == 503:
                # Modèle en cours de "cold start" côté Hugging Face — on
                # attend le temps estimé puis on retente.
                wait = 5
                try:
                    wait = float(resp.json().get("estimated_time", 5))
                except Exception:
                    pass
                time.sleep(min(wait, 15))
                continue

            print("[NLLB ERROR]", resp.status_code, resp.text[:300])
            return None

        except Exception as e:
            print("[NLLB EXCEPTION]", e)
            return None

    return None


def _apply_learned_corrections(source_text: str, translated_text: str, direction: str) -> str:
    """
    NLLB ne connaît pas les mots que l'utilisateur a enseignés en
    conversation. Si l'un de ces mots apparaît dans le texte source, on
    passe par une petite correction LLM ciblée pour s'assurer qu'il est
    bien respecté — sinon on ne touche à rien (pas d'appel inutile).
    """
    learned = _load_learned_glossary()
    if not learned:
        return translated_text

    lowered_source = source_text.lower()
    relevant = [
        v for k, v in learned.items()
        if re.search(rf"\b{re.escape(k)}\b", lowered_source)
    ]
    if not relevant:
        return translated_text

    pairs_text = "\n".join(f"- {v['wolof']} = {v['french']}" for v in relevant)
    lang_note = "wolof vers français" if direction == "wl_to_fr" else "français vers wolof"
    instruction = (
        f"Voici une traduction {lang_note} produite par un modèle de "
        "traduction automatique (NLLB). Vérifie qu'elle respecte "
        "STRICTEMENT ces correspondances apprises auprès de "
        f"l'utilisateur, et corrige la traduction si besoin :\n{pairs_text}\n\n"
        "Réponds UNIQUEMENT avec la traduction corrigée (identique si "
        "elle est déjà correcte), sans commentaire, sans guillemets."
    )
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": instruction},
                {
                    "role": "user",
                    "content": f"Texte source : {source_text}\nTraduction à vérifier : {translated_text}",
                },
            ],
            temperature=0,
            max_tokens=800,
        )
        corrected = (r.choices[0].message.content or "").strip()
        return corrected if corrected else translated_text
    except Exception as e:
        print("[GLOSSARY CORRECTION ERROR]", e)
        return translated_text


def _llm_translate_fallback(text: str, direction: str) -> str:
    """
    Repli utilisé quand NLLB est indisponible (pas de clé HF, panne,
    modèle indisponible) : traduction via Llama guidée par le glossaire.
    Moins fiable que NLLB pour le wolof, mais mieux que rien.
    """
    if direction == "wl_to_fr":
        instruction = (
            "Tu es un traducteur wolof → français. Traduis fidèlement le "
            "texte suivant en français, en t'appuyant en PRIORITÉ sur le "
            "glossaire fourni pour les termes qu'il couvre. Si un mot ne "
            "figure pas dans le glossaire et que tu n'es pas sûr de son "
            "sens, garde-le tel quel entre crochets (ex: [mot]) plutôt "
            "que d'inventer une traduction — mieux vaut un trou visible "
            "qu'un contresens. Réponds UNIQUEMENT avec la traduction, "
            "sans commentaire, sans guillemets, sans répéter le texte "
            "source, et ne dis jamais que tu ne comprends pas."
        )
    else:
        instruction = (
            "Tu es un traducteur français → wolof. Traduis fidèlement le "
            "texte suivant en wolof, en t'appuyant EN PRIORITÉ sur le "
            "glossaire fourni pour les termes qu'il couvre, et en gardant "
            "un wolof naturel et cohérent (n'invente pas de mots, garde "
            "en français les termes techniques/noms propres qui n'ont "
            "pas d'équivalent connu). Réponds UNIQUEMENT avec la "
            "traduction, sans commentaire, sans guillemets, sans répéter "
            "le texte source."
        )

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": f"{instruction}\n\n{get_full_glossary_text()}"},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        translated = (r.choices[0].message.content or "").strip()
        return translated if translated else text
    except Exception as e:
        print("[TRANSLATE ERROR]", direction, e)
        return text


def translate_text(text: str, direction: str):
    """
    Traduit `text`. direction : "wl_to_fr" ou "fr_to_wl".

    Priorité 1 : NLLB-200 (Hugging Face) — modèle dédié à la traduction,
    qui supporte réellement le wolof, donc bien plus fiable que de
    demander à un LLM généraliste de traduire.
    Priorité 2 (repli) : Llama + glossaire, si NLLB est indisponible.

    Dans les deux cas, si l'utilisateur a enseigné des mots en
    conversation et qu'ils apparaissent dans le texte source, une
    correction ciblée est appliquée pour les respecter.

    Retourne le texte traduit, ou le texte original en dernier recours
    (fail-open : mieux vaut traiter le texte original que planter).
    """
    if not text or not text.strip():
        return text

    src, tgt = ("wl", "fr") if direction == "wl_to_fr" else ("fr", "wl")

    nllb_result = _nllb_translate(text, src, tgt)
    if nllb_result:
        return _apply_learned_corrections(text, nllb_result, direction)

    return _llm_translate_fallback(text, direction)

# ─────────────────────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str):
    full_prompt = TYPE_PROMPTS.get(gen_type, "") + prompt
    encoded = urllib.parse.quote(full_prompt)

    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024"

    try:
        res = requests.get(url, timeout=60)
        return base64.b64encode(res.content).decode()
    except Exception as e:
        print("[IMAGE ERROR]", e)
        return None

# ─────────────────────────────────────────────
# IMAGE VISION (analyse d'une image envoyée par l'utilisateur)
# ─────────────────────────────────────────────
def _detect_image_mime(raw_bytes: bytes) -> str:
    """Détecte le type MIME réel à partir des premiers octets (signature de fichier)."""
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    # Par défaut : jpeg (format le plus courant depuis les galeries mobiles)
    return "image/jpeg"


def analyze_image_base64(image_base64: str, question: str):
    """
    Envoie l'image (base64) + une question à un modèle vision via Groq
    (Llama 4 Scout) et retourne la réponse texte du modèle, ou None en
    cas d'échec.
    """
    try:
        raw_bytes = base64.b64decode(image_base64)
        mime = _detect_image_mime(raw_bytes)
        data_url = f"data:{mime};base64,{image_base64}"
        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.5,
            max_tokens=600,
        )
        return r.choices[0].message.content
    except Exception as e:
        print("[VISION ERROR]", e)
        return None

# ─────────────────────────────────────────────
# DOCUMENT (extraction + analyse de PDF)
# ─────────────────────────────────────────────
MAX_DOC_CHARS = 15000  # limite de texte envoyée au LLM pour rester dans le contexte

def extract_pdf_text(pdf_base64: str):
    """
    Décode un PDF en base64 et en extrait le texte (toutes pages, tronqué
    si trop long). Retourne (texte, erreur) ; texte est None si l'extraction
    échoue (PDF scanné sans texte, fichier corrompu, etc.).
    """
    try:
        raw_bytes = base64.b64decode(pdf_base64)
        reader = PdfReader(io.BytesIO(raw_bytes))

        if reader.is_encrypted:
            return None, "Le PDF est protégé par mot de passe."

        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                continue

        full_text = "\n\n".join(t for t in pages_text if t.strip())

        if not full_text.strip():
            return None, "Aucun texte détecté (le PDF est probablement une image scannée)."

        if len(full_text) > MAX_DOC_CHARS:
            full_text = full_text[:MAX_DOC_CHARS] + "\n\n[...document tronqué, trop long...]"

        return full_text, None

    except Exception as e:
        print("[PDF ERROR]", e)
        return None, f"Impossible de lire ce PDF : {e}"


def analyze_document(doc_text: str, question: str):
    """
    Envoie le texte extrait du document + la question de l'utilisateur au
    LLM. Si la question est en wolof, on la traduit en français avant de
    l'envoyer (le document lui-même reste en français : c'est la langue
    dans laquelle le modèle raisonne le mieux), puis on retraduit la
    réponse en wolof.
    """
    is_wolof = detect_wolof(question)
    question_fr = translate_text(question, "wl_to_fr") if is_wolof else question

    try:
        prompt = (
            "Voici le contenu d'un document fourni par l'utilisateur :\n\n"
            f"---\n{doc_text}\n---\n\n"
            f"Question de l'utilisateur : {question_fr}"
        )
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=700,
        )
        answer = r.choices[0].message.content
    except Exception as e:
        print("[DOC ANALYSIS ERROR]", e)
        return None

    if is_wolof and answer:
        answer = translate_text(answer, "fr_to_wl")

    return answer

# ─────────────────────────────────────────────
# TEXT TO SPEECH
# ─────────────────────────────────────────────
def _edge_tts_sync(text: str, voice: str, out_path: str):
    """Wrapper synchrone pour edge_tts (lib asyncio)."""
    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)
    asyncio.run(_run())


def text_to_speech_base64(text: str, lang: str = "fr", max_retries: int = 2):
    """
    Génère un mp3 de la réponse.

    Priorité 1 : edge-tts — s'appuie sur l'infrastructure officielle de
    synthèse vocale de Microsoft Edge (Read Aloud), beaucoup plus stable
    en environnement serveur/cloud que gTTS.

    Priorité 2 (fallback) : gTTS — endpoint non-officiel de Google
    Translate ; peut renvoyer 403/429 selon l'IP sortante de l'hébergeur
    (observé sur certaines IP partagées de type Render).

    NB voix : ni edge-tts ni gTTS n'ont de voix wolof dédiée. On utilise la
    voix française même pour du texte wolof (prononciation approximative,
    mais reste compréhensible) — c'est une limitation connue, indépendante
    du problème de qualité texte traité ici.

    Retourne (audio_base64, message_erreur). message_erreur est None en
    cas de succès, sinon contient le détail des deux échecs pour debug
    direct dans les logs serveur / réponse JSON.
    """
    voice = "fr-FR-DeniseNeural"
    errors = []

    # ── Tentative 1 : edge-tts ──
    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            _edge_tts_sync(text, voice, tmp_path)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            if not audio_bytes:
                raise ValueError("fichier audio vide généré par edge-tts")

            return base64.b64encode(audio_bytes).decode(), None

        except Exception as e:
            err = f"edge-tts tentative {attempt + 1}: {type(e).__name__}: {e}"
            print("[TTS ERROR]", err)
            errors.append(err)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ── Tentative 2 (fallback) : gTTS ──
    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts = gTTS(text=text, lang=lang)
            tts.save(tmp_path)

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()

            if not audio_bytes:
                raise ValueError("fichier audio vide généré par gTTS")

            return base64.b64encode(audio_bytes).decode(), None

        except Exception as e:
            err = f"gTTS tentative {attempt + 1}: {type(e).__name__}: {e}"
            print("[TTS ERROR]", err)
            errors.append(err)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return None, " | ".join(errors)

# ─────────────────────────────────────────────
# SPEECH TO TEXT (Whisper via Groq)
# ─────────────────────────────────────────────
def transcribe_audio_base64(audio_base64: str):
    """
    Décode l'audio reçu en base64 (m4a/webm depuis le mobile) et le transcrit
    avec Whisper (Groq). Retourne le texte transcrit ou None en cas d'échec.
    """
    tmp_path = None
    try:
        audio_bytes = base64.b64decode(audio_base64)

        # On écrit en .m4a (format envoyé par l'app mobile iOS/Android).
        # Whisper/Groq se base sur le contenu réel du fichier, l'extension
        # sert surtout à l'API pour deviner le type — m4a est accepté.
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )

        # Le SDK Groq peut renvoyer soit une string, soit un objet avec .text
        text = transcription if isinstance(transcription, str) else getattr(transcription, "text", "")
        return text.strip() if text else None

    except Exception as e:
        print("[STT ERROR]", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

CONFUSION_MARKERS = [
    "je ne comprends pas", "je ne comprend pas", "je n'ai pas compris",
    "je ne saisis pas", "pourriez-vous reformuler", "peux-tu reformuler",
    "pouvez-vous préciser", "peux-tu préciser", "qu'est-ce que vous voulez dire",
    "qu'est-ce que tu veux dire", "je ne sais pas ce que", "pas sûr de comprendre",
]


def _looks_confused(text: str) -> bool:
    lowered = (text or "").lower()
    return any(m in lowered for m in CONFUSION_MARKERS)


def _run_chat_completion(working_message: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-10:]
    messages.append({"role": "user", "content": working_message})
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )
    return r.choices[0].message.content


# ─────────────────────────────────────────────
# CHAT HANDLER
# ─────────────────────────────────────────────
def handle_chat(user_message: str, history: list, want_audio_response: bool = False):
    intent = detect_image_intent(user_message)

    if intent:
        img = generate_image(intent["visual_prompt"], intent["type"])
        return {
            "response": intent["confirmation_message"],
            "has_image": True,
            "image_base64": img,
            "image_type": intent["type"],
            "visual_prompt": intent["visual_prompt"],
        }

    # ── Cas "enseignement" : l'utilisateur apprend un mot/une expression
    # wolof au bot ("Jamm veut dire paix"). On l'extrait et on le sauvegarde
    # dans le glossaire persistant, au lieu de laisser partir le message
    # vers le chat normal (où il finit souvent en "je ne comprends pas").
    if looks_like_teaching(user_message):
        pairs = extract_taught_pairs(user_message)
        added = add_learned_words(pairs)
        if added:
            lines = "\n".join(f"• « {p['wolof']} » = « {p['french']} »" for p in added)
            confirmation = (
                f"✅ Merci, j'ai retenu :\n{lines}\n\n"
                "Je m'en servirai dans nos prochaines conversations."
            )
            result = {"response": confirmation, "learned": added}
            if want_audio_response:
                audio_b64, tts_error = text_to_speech_base64(confirmation)
                result["audio_base64"] = audio_b64
                if tts_error:
                    result["tts_error"] = tts_error
            return result
        # Le message ressemblait à un enseignement mais rien d'exploitable
        # n'a été extrait (ex: LLM indécis) → on retombe sur le flux normal.

    # ── Traduction "sandwich" pour le wolof ──
    # On traite le message en français (langue que le modèle maîtrise
    # bien), puis on retraduit la réponse en wolof si besoin. C'est plus
    # fiable qu'une génération directe en wolof, et ça évite les
    # contresens dus à une mauvaise compréhension du wolof en entrée.
    is_wolof = detect_wolof(user_message)
    working_message = translate_text(user_message, "wl_to_fr") if is_wolof else user_message

    response_text = _run_chat_completion(working_message, history)

    # ── Filet de sécurité : si le message n'a pas été détecté comme wolof
    # (mot absent des marqueurs/glossaire) mais que le modèle répond par
    # une formule d'incompréhension, on retente en forçant le chemin de
    # traduction wolof → français avant de répondre. Ça rattrape les mots
    # wolof que l'heuristique locale n'a pas reconnus.
    if not is_wolof and _looks_confused(response_text):
        retried_message = translate_text(user_message, "wl_to_fr")
        if retried_message.strip().lower() != working_message.strip().lower():
            is_wolof = True
            response_text = _run_chat_completion(retried_message, history)

    if is_wolof and response_text:
        response_text = translate_text(response_text, "fr_to_wl")

    result = {"response": response_text}

    # On ne génère l'audio de réponse que si le client le demande
    # (ex: l'utilisateur a envoyé un vocal) pour ne pas surcharger
    # inutilement les requêtes texte classiques si besoin de couper ce comportement.
    if want_audio_response:
        audio_b64, tts_error = text_to_speech_base64(response_text)
        result["audio_base64"] = audio_b64
        if tts_error:
            # Visible dans les logs Render ET dans la réponse, pour debug rapide.
            print("[TTS] échec définitif après retries :", tts_error)
            result["tts_error"] = tts_error

    return result

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "pong"

@app.route("/tts", methods=["POST"])
def tts():
    """
    Génère l'audio d'un texte à la demande (bouton "écouter" sur un message
    bot déjà affiché). Le texte est fourni par le client — pas besoin de
    repasser par le LLM, on synthétise directement.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "texte manquant"}), 400

    if len(text) > 4000:
        text = text[:4000]

    audio_b64, tts_error = text_to_speech_base64(text)

    if not audio_b64:
        return jsonify({"error": tts_error or "échec de la synthèse vocale"}), 502

    return jsonify({"audio_base64": audio_b64})

@app.route("/glossary", methods=["GET"])
def glossary():
    """Debug : liste des mots wolof appris en conversation."""
    return jsonify({"learned": list(_load_learned_glossary().values())})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    has_audio = bool(data.get("has_audio"))
    audio_base64 = data.get("audio_base64")
    has_image = bool(data.get("has_image"))
    image_base64 = data.get("image_base64")
    has_document = bool(data.get("has_document"))
    document_base64 = data.get("document_base64")
    history = data.get("history", [])

    # ── Cas document : extraction texte + analyse, pas besoin de passer par handle_chat ──
    if has_document:
        if not document_base64:
            return jsonify({"error": "document manquant"}), 400

        doc_text, doc_error = extract_pdf_text(document_base64)

        if not doc_text:
            return jsonify({
                "error": doc_error or "Impossible de lire ce document",
                "response": f"❌ {doc_error or 'Je n’ai pas pu lire ce document.'}",
            }), 200

        question = (data.get("message") or "Résume ce document en français.").strip()
        response_text = analyze_document(doc_text, question)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser le document",
                "response": "❌ Je n'ai pas réussi à analyser ce document, réessaie.",
            }), 200

        return jsonify({"response": response_text})

    # ── Cas image : analyse vision directe, pas besoin de passer par handle_chat ──
    if has_image:
        if not image_base64:
            return jsonify({"error": "image manquante"}), 400

        question = (data.get("message") or "Décris cette image en détail en français.").strip()
        response_text = analyze_image_base64(image_base64, question)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser l'image",
                "response": "❌ Je n'ai pas réussi à analyser cette image, réessaie.",
            }), 200

        return jsonify({"response": response_text})

    transcription = None

    if has_audio:
        if not audio_base64:
            return jsonify({"error": "audio manquant"}), 400

        transcription = transcribe_audio_base64(audio_base64)

        if not transcription:
            return jsonify({
                "error": "Impossible de transcrire l'audio",
                "response": "❌ Je n'ai pas réussi à comprendre le message vocal, réessaie.",
            }), 200

        user_message = transcription
    else:
        user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"error": "empty message"}), 400

    try:
        result = handle_chat(user_message, history, want_audio_response=has_audio)
        if transcription:
            result["transcription"] = transcription
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Yelen AI API 🌟"

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
