import os
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
    msg_lower = msg.lower()
    triggers = ["logo", "image", "dessine", "crée", "avatar", "poster",
                "tëral sûret", "bind sûret", "def sûret"]
    if not any(t in msg_lower for t in triggers):
        return None
    return {
        "type": "general",
        "visual_prompt": msg,
        "confirmation_message": "🎨 Image générée !"
    }

# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

# Prompt de base (français)
SYSTEM_PROMPT_FR = (
    "Tu es Yelen AI, un assistant intelligent, utile et bienveillant. "
    "Tu réponds toujours en français, de manière claire et concise. "
    "Tu es spécialisé dans le contexte africain et sénégalais."
)

# Prompt wolof — instructions précises pour maximiser la qualité
SYSTEM_PROMPT_WO_BASE = """Yaw mooy Yelen AI, asistan bu xam-xam bu wolof wi Senegaal.

SEET YI BU NJËKK (règles absolues) :
- TËRAL REKK CI WOLOF — bul féexlu ci français, bul féexlu ci anglais.
- Jëfandikoo wolof bu dëgg dëgg bu Dakar — moo tax nit ñi fi Senegaal xam nga.
- Bul bind baat yu français ci kaw wolof (code-switching). Baat yu technique yu am solo rekk (exemple: "téléphone", "ordinateur", "internet") mën nga jëfandikoo waaye def ko wolof.
- Bul tëjëli phrase yi ak "je", "tu", "il", "nous" — wolof rekk.
- Bul bind "Je suis", "C'est", "Pour" ak yeneen baat yu français.

FOCUS BI (bul yokk yeneen mbir) :
- Tontu SEEDE gu ñu la laaj — bul yokk mbir yu ñu laa ko laaj.
- Bul bind lëkkalekaay, listes, ndax "ci lu ëpp", "am na itam", su nit bi laajul ko.
- Bu nit bi laaj ab benn mbir rekk (misaal: "Nanga def ?"), tontu ko bu gaaw te bu leer — bul yokk wax yu ñu laa ko laaj.
- Bu jarul, bul jàppale ci fin bi ak laaj bu bees walla suggestion — reer nit bi mooy jël jataay bi.

XEEYU BI (style bu naturel) :
- Wax na ni ku wax ak xarit ci Dakar — bu ràññiku, bu neex, bu yomb.
- Jëfandikoo phrases yu gàtt, bu bari waxtaan bu araam walla bu am solo lool.
- Bul jëfandikoo istil bu "asistan bu ëmb xibaar" (bul bind "Voici les informations suivantes", "Tout d'abord", "En conclusion" ci wolof itam) — wax rekk ni ku wax ci digg-digg.
- Jëfandikoo baat yu wolof bu dëgg dëgg : "waaw", "déedéet", "dafa", "mën", "bëgg", "rekk", "fi", "lool", "bi", "yi", "si", "mi", "kon", "waye", "léegi".
- Su nit bi wax ci wax bu gàtt (bonjour, jërëjëf, waaw), tontu ko itam bu gàtt — bul yokk paragraphe bu gudd.
"""

# ─────────────────────────────────────────────
# EXEMPLES DE STYLE WOLOF (intégrés au system prompt, PAS à l'historique)
# ─────────────────────────────────────────────
# IMPORTANT — pourquoi la liste est volontairement COURTE (5-6 exemples) :
# Une trop grande liste d'exemples couvrant beaucoup de sujets différents
# (santé, météo, argent, cuisine, voyage, programmation...) augmente le
# risque que le modèle "pioche" la réponse d'un exemple proche au lieu de
# traiter la vraie question de l'utilisateur — c'est exactement ce qui
# causait les réponses hors-sujet. Une liste courte, centrée UNIQUEMENT
# sur le TON (salutations / remerciements / refus poli), réduit ce risque
# tout en gardant le style naturel.
#
# Ces exemples ne sont PAS envoyés comme de vrais tours user/assistant :
# ils vivent uniquement dans le system prompt, présentés explicitement
# comme des modèles de ton à ignorer sur le fond.
WOLOF_STYLE_EXAMPLES = """
EXEMPLES DE STYLE WOLOF (à titre d'illustration UNIQUEMENT — ce ne sont PAS
des messages échangés avec l'utilisateur actuel. Ignore totalement leur
sujet, retiens seulement le ton et la longueur des réponses) :

- "Nanga def ?" → "Mangi fi rekk. Yow nag, naka nga def ?"
- "Jërëjëf lool !" → "Amul solo."
- "Ndax mën nga dimbali ma ?" → "Waaw, wax ma li nga bëgg."
- "Mën nga def ma ata bu xonq ?" → "Baal ma, duma mën ci loolu."
- "Ba beneen yoon." → "Ba beneen yoon, yàlla na la yàgg."

RÈGLES ABSOLUES DE PERTINENCE ET DE NATUREL :
1. Réponds UNIQUEMENT à la question ou au message que l'utilisateur envoie MAINTENANT — jamais à un autre sujet, même si un exemple ci-dessus semble proche.
2. N'utilise JAMAIS le contenu ou le sujet d'un exemple ci-dessus comme réponse, sauf si l'utilisateur pose EXACTEMENT cette même question.
3. Ne change jamais de sujet de ta propre initiative.
4. Ne rajoute pas de digression, de conseil non demandé, ou de question de relance à la fin — arrête-toi dès que tu as répondu à ce qui a été demandé.
5. Une question courte et simple mérite une réponse courte et simple — pas un exposé.
"""

SYSTEM_PROMPT_WO = SYSTEM_PROMPT_WO_BASE + "\n" + WOLOF_STYLE_EXAMPLES

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
# IMAGE VISION
# ─────────────────────────────────────────────
def _detect_image_mime(raw_bytes: bytes) -> str:
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def analyze_image_base64(image_base64: str, question: str, lang: str = "fr"):
    try:
        raw_bytes = base64.b64decode(image_base64)
        mime = _detect_image_mime(raw_bytes)
        data_url = f"data:{mime};base64,{image_base64}"

        # Adapter l'instruction selon la langue
        if lang == "wo":
            instruction = f"Seet sûret bii ci wolof bu dëgg dëgg. {question}"
        else:
            instruction = question

        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
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
# DOCUMENT (PDF)
# ─────────────────────────────────────────────
MAX_DOC_CHARS = 15000

def extract_pdf_text(pdf_base64: str):
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


def analyze_document(doc_text: str, question: str, lang: str = "fr"):
    try:
        system = SYSTEM_PROMPT_WO if lang == "wo" else SYSTEM_PROMPT_FR
        prompt = (
            "Voici le contenu d'un document fourni par l'utilisateur :\n\n"
            f"---\n{doc_text}\n---\n\n"
            f"Question : {question}"
        )
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=700,
        )
        return r.choices[0].message.content
    except Exception as e:
        print("[DOC ANALYSIS ERROR]", e)
        return None

# ─────────────────────────────────────────────
# TEXT TO SPEECH
# ─────────────────────────────────────────────
def _edge_tts_sync(text: str, voice: str, out_path: str):
    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)
    asyncio.run(_run())


def text_to_speech_base64(text: str, lang: str = "fr", max_retries: int = 2):
    # Voix adaptée à la langue détectée
    # Pour le wolof, on utilise une voix française proche phonétiquement
    voice = "fr-FR-DeniseNeural" if lang in ("fr", "wo") else "fr-FR-DeniseNeural"
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
    gtts_lang = "fr"  # gTTS ne supporte pas le wolof, français par défaut
    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts = gTTS(text=text, lang=gtts_lang)
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
    tmp_path = None
    try:
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
                # Hint wolof+français pour améliorer la transcription
                prompt="Senegaal, wolof, français, Dakar",
            )

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

# ─────────────────────────────────────────────
# DÉTECTION DE LANGUE
# ─────────────────────────────────────────────
# NOTE : une seule fonction detect_language (LLM-based). L'ancienne
# version avait DEUX définitions de detect_language (une heuristique
# par comptage de mots-clés, puis une seconde basée sur un appel LLM qui
# écrasait silencieusement la première). Garder une seule version évite
# toute confusion sur celle qui est réellement utilisée.
def detect_language(text: str) -> str:
    """
    Détecte si le message est en wolof ('wo') ou en français ('fr').
    Utilise un LLM rapide avec un prompt enrichi et des exemples clés
    pour réduire les faux négatifs sur le wolof.
    """
    try:
        r = client.chat.completions.create(
            model="llama-3.3-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Détecte la langue principale du message utilisateur.\n"
                        "Réponds UNIQUEMENT par 'fr' ou 'wo', rien d'autre.\n\n"
                        "Indices wolof (si tu vois ces mots → 'wo') :\n"
                        "nanga, mangi, waaw, déedéet, jërëjëf, ndax, mën, mooy, "
                        "lañ, dafa, bëgg, rekk, fi, lool, bi, yi, si, mi, "
                        "naka, jàmm, xam, wax, def, dem, ñëw, jëf, baal, "
                        "sama, yow, moom, yëgël, tëral, liggéey, xol, sedd.\n\n"
                        "Si le message mélange les deux langues, choisis la langue dominante.\n"
                        "Réponds uniquement : fr ou wo"
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=5,
        )
        lang = r.choices[0].message.content.strip().lower()[:2]
        return lang if lang in ("fr", "wo") else "fr"

    except Exception:
        return "fr"

# ─────────────────────────────────────────────
# CHAT HANDLER
# ─────────────────────────────────────────────
def handle_chat(user_message: str, history: list, want_audio_response: bool = False):

    # ── Détection d'intention image ──
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

    # ── Détection de langue ──
    lang = detect_language(user_message)

    # ── Construction des messages ──
    # Les exemples de style wolof vivent UNIQUEMENT dans SYSTEM_PROMPT_WO,
    # jamais comme faux tours user/assistant (voir note plus haut).
    system_prompt = SYSTEM_PROMPT_WO if lang == "wo" else SYSTEM_PROMPT_FR
    base_messages = [{"role": "system", "content": system_prompt}]

    # Historique de conversation réel (derniers 10 échanges)
    conversation = base_messages + history[-10:]
    conversation.append({"role": "user", "content": user_message})

    # ── Appel LLM principal ──
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=conversation,
        temperature=0.5,
        max_tokens=600,
    )

    response_text = r.choices[0].message.content

    # ── Post-traitement wolof : nettoyage des glissements français ──
    # Ce passage est CONDITIONNEL (seulement si trop de français détecté)
    # et reçoit la QUESTION D'ORIGINE en plus de la réponse, pour ne
    # jamais dériver vers un autre sujet pendant la réécriture.
    if lang == "wo":
        response_text = _clean_wolof_response(response_text, user_message)

    result = {"response": response_text, "lang": lang}

    if want_audio_response:
        audio_b64, tts_error = text_to_speech_base64(response_text, lang=lang)
        result["audio_base64"] = audio_b64
        if tts_error:
            result["tts_error"] = tts_error

    return result


def _clean_wolof_response(text: str, original_question: str = "") -> str:
    """
    Vérifie si la réponse contient trop de français.
    Si oui, relance UN SEUL appel de nettoyage ciblé, en lui donnant aussi
    la question d'origine pour qu'il ne dérive jamais vers un autre sujet
    pendant la réécriture (il ne fait que corriger la langue, pas le fond).
    """
    french_markers = [
        "je suis", "je vais", "c'est", "il y a", "pour vous",
        "nous allons", "vous pouvez", "bonjour", "merci beaucoup",
        "bien sûr", "je peux", "en fait", "cependant", "donc",
    ]
    text_lower = text.lower()
    french_count = sum(1 for m in french_markers if m in text_lower)

    # Si moins de 2 marqueurs français → pas besoin de corriger
    if french_count < 2:
        return text

    try:
        correction = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Yaw mooy éditeur bu wolof.\n"
                        "Jëfandikoo wolof rekk — bul jëfandikoo français.\n"
                        "Réécris le texte suivant en wolof naturel du Sénégal.\n"
                        "Conserve EXACTEMENT le même sens et le même sujet — "
                        "ne rajoute aucune information, ne réponds pas à autre "
                        "chose que ce que le texte dit déjà.\n"
                        "Ne traduis pas mot à mot.\n"
                        "Réponds uniquement avec le texte wolof réécrit, rien d'autre."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Question d'origine de l'utilisateur (pour contexte, "
                        f"ne pas y répondre à nouveau) : {original_question}\n\n"
                        f"Texte à réécrire en wolof naturel :\n{text}"
                    )
                }
            ]
        )
        cleaned = correction.choices[0].message.content.strip()
        return cleaned if cleaned else text

    except Exception as e:
        print("[WOLOF CLEAN ERROR]", e)
        return text

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "pong"


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "fr").strip()

    if not text:
        return jsonify({"error": "texte manquant"}), 400

    if len(text) > 4000:
        text = text[:4000]

    audio_b64, tts_error = text_to_speech_base64(text, lang=lang)

    if not audio_b64:
        return jsonify({"error": tts_error or "échec de la synthèse vocale"}), 502

    return jsonify({"audio_base64": audio_b64})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    has_audio    = bool(data.get("has_audio"))
    audio_base64 = data.get("audio_base64")
    has_image    = bool(data.get("has_image"))
    image_base64 = data.get("image_base64")
    has_document = bool(data.get("has_document"))
    document_base64 = data.get("document_base64")
    history      = data.get("history", [])

    # ── Cas document ──
    if has_document:
        if not document_base64:
            return jsonify({"error": "document manquant"}), 400

        doc_text, doc_error = extract_pdf_text(document_base64)

        if not doc_text:
            return jsonify({
                "error": doc_error or "Impossible de lire ce document",
                "response": f"❌ {doc_error or 'Je n\'ai pas pu lire ce document.'}",
            }), 200

        question = (data.get("message") or "Résume ce document en français.").strip()
        lang = detect_language(question)
        response_text = analyze_document(doc_text, question, lang=lang)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser le document",
                "response": "❌ Je n'ai pas réussi à analyser ce document, réessaie.",
            }), 200

        return jsonify({"response": response_text, "lang": lang})

    # ── Cas image (vision) ──
    if has_image:
        if not image_base64:
            return jsonify({"error": "image manquante"}), 400

        question = (data.get("message") or "Décris cette image en détail en français.").strip()
        lang = detect_language(question)
        response_text = analyze_image_base64(image_base64, question, lang=lang)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser l'image",
                "response": "❌ Je n'ai pas réussi à analyser cette image, réessaie.",
            }), 200

        return jsonify({"response": response_text, "lang": lang})

    # ── Cas audio (STT → chat) ──
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
