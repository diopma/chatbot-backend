import os
import base64
import tempfile
import urllib.parse
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY manquant")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────
# IMAGE STYLES & SIZES
# ─────────────────────────────────────────────────────────────
TYPE_PROMPTS = {
    "logo":         "minimalist professional flat vector logo, clean sharp edges, bold colors, white background, centered composition, SVG style",
    "icon":         "simple modern app icon, flat design, bold colors, clean lines, centered, rounded corners",
    "illustration": "vibrant detailed african art illustration, rich warm colors, professional digital painting, intricate patterns",
    "photo":        "photorealistic professional DSLR photo, sharp focus, perfect lighting, 8k resolution, high detail",
    "pattern":      "beautiful seamless african kente textile pattern, vibrant traditional colors, intricate geometric weave",
    "banner":       "modern professional wide marketing banner, bold typography, vibrant gradient colors, eye-catching",
    "avatar":       "professional portrait photo, centered face, studio lighting, sharp focus, neutral background",
    "poster":       "eye-catching event poster design, bold typography, vibrant colors, dynamic composition",
    "general":      "high quality professional digital art, vibrant colors, sharp details, 4k, masterpiece",
}

TYPE_SIZES = {
    "logo":         (1024, 1024),
    "icon":         (512,  512),
    "illustration": (768,  1024),
    "photo":        (1024, 1024),
    "pattern":      (1024, 1024),
    "banner":       (1024, 512),
    "avatar":       (512,  512),
    "poster":       (768,  1024),
    "general":      (1024, 1024),
}

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo"],
    "icon":         ["icône", "icone", "icon"],
    "illustration": ["illustration", "art", "dessin", "nataal bu", "nataal yu rafet"],
    "photo":        ["photo", "photographie", "réaliste", "realistic", "nataal bu dëkk"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu", "mbañ"],
    "banner":       ["bannière", "banniere", "banner", "couverture", "nataal bu bon"],
    "avatar":       ["avatar", "profil", "portrait", "visage", "seen bët", "sama bët"],
    "poster":       ["affiche", "poster", "flyer", "nataal bu liggéey"],
}

# ─────────────────────────────────────────────────────────────
# DÉTECTION IMAGE — mots déclencheurs wolof + français
# ─────────────────────────────────────────────────────────────
IMAGE_TRIGGERS = [
    # Français
    "logo", "logos", "icône", "icones", "icon", "icons",
    "illustration", "illustrations", "avatar", "avatars",
    "bannière", "banniere", "banner", "banners",
    "affiche", "poster", "posters", "flyer", "flyers",
    "motif", "pattern", "patterns", "visuel", "visuels",
    "dessin", "dessins", "portrait", "portraits",
    # Wolof — image / photo
    "nataal", "nataalu", "nataalyi", "nataalye",
    "sama nataal", "seen nataal", "yenn nataal",
    "nataal bu rafet", "nataal bu baax",
    "liggéey bu nataal", "liggeeyu nataal",
]

IMAGE_VERBS = [
    # Français
    "génère", "générer", "genere", "generer",
    "crée", "créer", "cree", "creer",
    "dessine", "dessiner", "fais", "faire",
    "montre", "montrer", "produis", "produire",
    "réalise", "realise", "imagine",
    "génère-moi", "fais-moi", "crée-moi",
    # Anglais
    "generate", "create", "draw", "make", "render", "design",
    # Wolof — verbes d'action
    "def", "defal", "deflu",          # faire / créer
    "bind", "bindal", "bindaale",     # dessiner / écrire
    "yëgël", "yegal", "yëgëlal",     # montrer
    "def ma", "bind ma", "yëgël ma",  # fais-moi / dessine-moi / montre-moi
    "yokk", "yokkal",                  # ajouter / faire
    "am", "amal",                      # avoir / produire
    "teg", "tegal",                    # mettre / créer
    "daldi def", "daldi bind",         # vas créer / vas dessiner
    "seet", "seetal",                  # regarder / montrer
    "wone", "woneel",                  # montrer / présenter
]

def _norm(text: str) -> str:
    """Normalise le texte : minuscules + suppression accents."""
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a").replace("ç","c")
        .replace("ù","u").replace("û","u").replace("î","i")
        .replace("ï","i").replace("ô","o"))

def detect_image_intent(msg: str) -> dict | None:
    m     = _norm(msg)
    words = m.split()

    has_noun = any(
        _norm(n) in words or _norm(n) in m
        for n in IMAGE_TRIGGERS
    )
    has_verb = any(
        _norm(v) in words or _norm(v) in m
        for v in IMAGE_VERBS
    )

    # Déclenche si : (verbe + nom) OU nom seul suffit
    if not (has_noun or (has_verb and has_noun)):
        return None

    # Détecter le type d'image
    gen_type = "general"
    for t, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(_norm(kw) in m for kw in keywords):
            gen_type = t
            break

    return {
        "type":                 gen_type,
        "visual_prompt":        msg,
        "confirmation_message": "🎨 Image générée !",
    }

# ─────────────────────────────────────────────────────────────
# DÉTECTION LANGUE wolof vs français
# ─────────────────────────────────────────────────────────────
WOLOF_WORDS = {
    # ── Score 4 : expressions très spécifiques au wolof ──
    "nanga def": 4, "nanga xam": 4, "nanga dem": 4,
    "mangi fi rekk": 4, "maa ngi fi": 4, "mangi dem": 4,
    "jërejëf lool": 4, "baal ma ko": 4, "waaw waaw": 4,
    "dafa baax": 4, "dafa neex": 4, "dafa mel ni": 4,
    "xam naa": 4, "faamaak": 4, "amul solo": 4,
    "def ma": 4, "bind ma": 4, "yëgël ma": 4,
    "soo bëgg": 4, "bëgg naa": 4,
    "lu baax": 4, "lu neex": 4,

    # ── Score 3 : mots très spécifiques au wolof ──
    "jërejëf": 3, "jërëjëf": 3, "baal ma": 3,
    "deedeet": 3, "mangi fi": 3, "mangi": 3,
    "maa ngi": 3, "nataal": 3, "nataalu": 3,
    "liggéey": 3, "liggeeyu": 3,
    "dafa": 3, "dama": 3, "xam": 3,
    "bëgg": 3, "siiw": 3, "rekk": 3,
    "yëgël": 3, "woneel": 3, "wone": 3,
    "ndanka": 3, "ndanka ndanka": 3,
    "mbokk": 3, "xarit": 3,

    # ── Score 2 : fréquent en wolof ──
    "waaw": 2, "yow": 2, "moom": 2,
    "laa": 2, "naa": 2, "nga": 2, "niit": 2,
    "nekk": 2, "topp": 2, "wax": 2,
    "gis": 2, "gisul": 2,
    "dëkk": 2, "sunu": 2, "leen": 2,
    "rafet": 2, "baax": 2, "neex": 2,
    "ndax": 2, "waaye": 2, "mbaa": 2,
    "tey": 2, "bëccëk": 2, "guddi": 2,
    "jaay": 2, "jënd": 2,
    "xol": 2, "bàkkaar": 2,
    "daldi": 2, "seet": 2,

    # ── Score 1 : mots courants ──
    "dem": 1, "ñëw": 1, "lekk": 1, "dox": 1,
    "fëkk": 1, "bind": 1, "jëf": 1, "tëdd": 1,
    "ak": 1, "sama": 1, "seen": 1,
    "ci": 1, "bi": 1, "yi": 1, "bu": 1, "si": 1,
    "baay": 1, "yaay": 1, "xale": 1,
    "goor": 1, "jigéen": 1, "doom": 1,
    "benn": 1, "ñaar": 1, "ñett": 1,
    "ñent": 1, "juróom": 1, "fukk": 1,
    "lool": 1, "def": 1, "am": 1,
    "ko": 1, "leen": 1, "len": 1,
    "mu": 1, "nu": 1, "ñu": 1,
    "dox": 1, "set": 1, "setal": 1,
    "tëral": 1, "tax": 1, "di": 1,
}

def detect_language(text: str) -> str:
    t = text.lower()
    wolof_score = sum(w for m, w in WOLOF_WORDS.items() if m in t)

    french_words = [
        "je", "tu", "il", "elle", "nous", "vous",
        "les", "des", "une", "est", "avec", "bonjour",
        "merci", "comment", "pourquoi", "mais", "donc",
        "alors", "parce", "quand", "pour", "dans",
    ]
    french_score = sum(1 for w in french_words if f" {w} " in f" {t} ")

    print(f"[LANG SCORE] wolof={wolof_score} french={french_score}")

    if wolof_score >= 2:
        return "wolof"
    if wolof_score >= 1 and french_score >= 1:
        return "wolof"   # mélange → traiter comme wolof
    if french_score >= 2:
        return "french"
    return "french"      # défaut = français

# ─────────────────────────────────────────────────────────────
# TRADUCTION PROMPT → ANGLAIS (pour FLUX)
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an image prompt translator. "
                        "Translate the user's request (in any language) into a detailed "
                        "English image generation prompt. Be descriptive and specific. "
                        "Add quality keywords like 'professional', 'high quality', 'detailed'. "
                        "Return ONLY the English prompt, no explanation, no quotes."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=120,
        )
        translated = r.choices[0].message.content.strip()
        print(f"[PROMPT EN] {translated}")
        return translated
    except Exception as e:
        print(f"[TRANSLATE ERR] {e}")
        return prompt

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    english = translate_prompt_to_english(prompt)
    prefix  = TYPE_PROMPTS.get(gen_type, "")
    full    = f"{prefix}, {english}, masterpiece, best quality, ultra detailed, sharp"
    w, h    = TYPE_SIZES.get(gen_type, (1024, 1024))

    print(f"[IMAGE] type={gen_type} | {full[:80]}...")

    # Together AI — FLUX
    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":           "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt":          full,
                    "width":           min(w, 1024),
                    "height":          min(h, 1024),
                    "steps":           4,
                    "n":               1,
                    "response_format": "b64_json",
                },
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["data"][0]["b64_json"]
        except Exception as e:
            print("[FLUX ERR]", e)

    # Fallback Pollinations
    try:
        enc = urllib.parse.quote(full)
        url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&enhance=true&model=flux"
        res = requests.get(url, timeout=90)
        return base64.b64encode(res.content).decode()
    except Exception as e:
        print("[POLLINATIONS ERR]", e)
        return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION MIME
# ─────────────────────────────────────────────────────────────
def get_mime(b64: str) -> str:
    try:
        h = base64.b64decode(b64[:20])
        if h[:4] == b'\x89PNG':  return "image/png"
        if h[:2] == b'\xff\xd8': return "image/jpeg"
        if b'WEBP' in h:         return "image/webp"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION AUDIO — optimisée wolof + français
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    # Détection format
    suffix, mime = ".m4a", "audio/mp4"
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b'RIFF':
            suffix, mime = ".wav", "audio/wav"
        elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            suffix, mime = ".mp3", "audio/mpeg"
        elif len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            suffix, mime = ".m4a", "audio/mp4"
        elif audio_bytes[:4] == b'OggS':
            suffix, mime = ".ogg", "audio/ogg"

    print(f"[AUDIO] {suffix} — {len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio{suffix}", f, mime),
                response_format="text",          # ← "text" uniquement, Groq ne supporte pas verbose_json
                prompt=(
                    "Transcris exactement ce qui est dit. "
                    "Ce message est en français ou en wolof sénégalais, "
                    "ou un mélange des deux. "
                    "Mots wolof fréquents: nanga def, mangi fi, jërejëf, waaw, "
                    "deedeet, dama, dafa, xam, dem, ñëw, lekk, wax, nekk, "
                    "bi, yi, ci, ak, sama, bëgg, nataal, def ma, bind ma, "
                    "xale, baay, yaay, xarit, mbokk, ndax, baal ma, yow. "
                    "Termes tech possibles: logo, image, avatar, créer, générer."
                ),
            )
        # response_format="text" retourne directement une string
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        print(f"[WHISPER] texte={repr(text)}")
        return (text or "").strip()

    finally:
        os.unlink(path)

# ─────────────────────────────────────────────────────────────
# HANDLE CHAT — cœur de la logique
# ─────────────────────────────────────────────────────────────
WOLOF_SYSTEM = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles wolof et français couramment, comme un jeune Dakarois éduqué.

RÈGLES STRICTES :
1. Si l'utilisateur parle wolof → réponds EN WOLOF avec du français si nécessaire.
2. Si mélange wolof-français (nouchi dakarois) → réponds dans le même style.
3. Ne force PAS le wolof si la question est 100% française → réponds en français.
4. Sois concis, naturel, chaleureux. Pas de réponses trop longues.
5. N'invente JAMAIS de mots wolof inexistants — utilise du français si tu ne sais pas.
6. Pour générer une image : dis "Wax ko : 'def ma logo bu...' walla 'nataal...'"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOCABULAIRE WOLOF ESSENTIEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SALUTATIONS :
• "Nanga def ?" → Comment vas-tu ?
• "Mangi fi rekk" → Je suis là / Ça va
• "Nanga xam ?" → Est-ce que tu sais ?
• "Maa ngi dem" → Je m'en vais
• "Asalaa maalekum" → Paix sur vous
• "Maalekum salaam" → Et sur vous la paix

REMERCIEMENTS / POLITESSE :
• "Jërejëf" → Merci
• "Jërejëf lool" → Merci beaucoup
• "Baal ma" → Excuse-moi / Pardon
• "Amul solo" → Pas de problème
• "Yëgël na !" → Bien sûr ! / Je comprends !

OUI / NON :
• "Waaw" → Oui
• "Waaw waaw" → Oui oui (confirmer)
• "Deedeet" → Non
• "Siiw" → Exact / C'est ça

ÉMOTIONS / QUALITÉS :
• "Dafa baax" → C'est bien / C'est bon
• "Dafa neex" → C'est agréable / C'est bon (goût/sensation)
• "Dafa rafet" → C'est beau
• "Dafa mel ni..." → Ça ressemble à...
• "Neex na lool" → C'est vraiment bien
• "Baax na" → C'est bon / Ça va
• "Rafet na" → C'est beau

VERBES COURANTS :
• "Dem" → Aller
• "Ñëw" → Venir
• "Lekk" → Manger
• "Lekkal" → Mange (impératif)
• "Wax" → Parler / Dire
• "Xam" → Savoir / Connaître
• "Bëgg" → Vouloir / Aimer
• "Gis" → Voir
• "Jëf" → Faire (action)
• "Def" → Faire / Créer
• "Bind" → Écrire / Dessiner
• "Dox" → Marcher / Aller
• "Jaay" → Vendre
• "Jënd" → Acheter
• "Topp" → Suivre
• "Seet" → Regarder / Vérifier
• "Nekk" → Être / Se trouver
• "Am" → Avoir

PRONOMS :
• "Maa ngi" / "Mangi" → Je suis / Moi je
• "Dama" → Je (+ verbe)
• "Dafa" → Il/Elle (+ verbe)
• "Yow" → Toi
• "Moom" → Lui / Elle
• "Sunu" → Notre / Nous
• "Seen" → Leur / Eux
• "Sama" → Mon / Ma / Mes

TEMPS / MOMENT :
• "Tey" → Aujourd'hui
• "Bëccëk" → Ce matin
• "Guddi" → Ce soir / La nuit
• "Bi" → Le (article défini)
• "Yi" → Les (article pluriel)

QUESTIONS :
• "Lan ?" → Quoi ?
• "Kan ?" → Qui ?
• "Fan ?" → Où ?
• "Naka ?" → Comment ?
• "Ndax ?" → Est-ce que ? / Pourquoi ?
• "Buki ?" → Combien ?
• "Kañ ?" → Quand ?

EXPRESSIONS POPULAIRES DAKAR :
• "Dafa gëna baax" → C'est encore mieux
• "Ndanka ndanka" → Doucement / Petit à petit
• "Waaw, rekk !" → Oui, exactement !
• "Maa ko def" → Je vais le faire
• "Xam naa" → Je sais / Je comprends
• "Xamul" → Il/elle ne sait pas
• "Dëkk bi" → La ville / Le quartier
• "Bu baax" → Si c'est bien
• "Soo bëgg" → Si tu veux
• "Lu baax" → Ce qui est bien
• "Téere bi" → Le livre
• "Liggéey" → Travail / Travailler
• "Jaraay" → Se battre / Essayer
• "Daldi" → Vite / Allons-y

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXEMPLES DE CONVERSATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "Nanga def ?" → "Mangi fi rekk, jërejëf ! Yow noo ?"
• "Dama bëgg xam..." → "Waaw, maa ngi wax la ci. [réponse]. Xam naa ?"
• "Logo bi rafet na !" → "Jërejëf lool ! Dafa baax na ?"
• "Def ma yenn logo" → "Waaw ! Wax ma soo bëgg : magasin bi, restaurant, walla lan ?"
• "Lan moy Yelen AI ?" → "Maa ngi Yelen AI — intelligence artificielle bu Afrika. Dama dem jëf ci sa liggéey ak sa kow !"
• "Yaangi dem fan ?" → "Mangi nekk ci internet bi, dégëlu kaw !"
"""

FRENCH_SYSTEM = (
    "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
    "Tu réponds toujours en français. "
    "Tu peux créer des images, logos, illustrations — "
    "si demandé, utilise : 'crée un logo', 'génère une image', etc."
)

def handle_chat(user_message: str, history: list) -> dict:
    # 1. Vérifier si c'est une demande d'image
    intent = detect_image_intent(user_message)
    if intent:
        img = generate_image(intent["visual_prompt"], intent["type"])
        if img:
            return {
                "response":      intent["confirmation_message"],
                "has_image":     True,
                "image_base64":  img,
                "image_type":    intent["type"],
                "visual_prompt": intent["visual_prompt"],
            }
        return {"response": "❌ Génération échouée. Réessaie dans quelques secondes."}

    # 2. Détecter la langue
    lang = detect_language(user_message)
    system = WOLOF_SYSTEM if lang == "wolof" else FRENCH_SYSTEM

    # 3. Construire les messages
    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # 4. Appel LLM
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=500,
    )
    return {"response": r.choices[0].message.content}

# ─────────────────────────────────────────────────────────────
# KEEP-ALIVE
# ─────────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ─────────────────────────────────────────────────────────────
# ROUTE /chat
# ─────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    user_message = data.get("message", "")
    has_image    = data.get("has_image", False)
    image_base64 = data.get("image_base64")
    has_audio    = data.get("has_audio", False)
    audio_base64 = data.get("audio_base64")
    history      = data.get("history", [])

    # ── 🎙 AUDIO ──
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            if len(audio_bytes) < 500:
                return jsonify({"response": "❌ Audio trop court. Parle un peu plus longtemps."})

            transcribed = transcribe_audio(audio_bytes)
            print("[TRANSCRIPTION]", repr(transcribed))

            if not transcribed or len(transcribed.strip()) < 2:
                return jsonify({"response": "❌ Audio non reconnu. Rapproche-toi du micro et réessaie."})

            result = handle_chat(transcribed, history)
            result["transcription"] = transcribed
            return jsonify(result)

        except Exception as e:
            print("[AUDIO ERR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ── 🖼 ANALYSE IMAGE ──
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})

            mime     = get_mime(image_base64)
            question = user_message.strip() or "Décris cette image en détail en français."

            for model in [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
            ]:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}},
                            {"type": "text", "text": question},
                        ]}],
                        max_tokens=1024,
                    )
                    return jsonify({"response": r.choices[0].message.content})
                except Exception as e:
                    print(f"[IMG ERR] {model}: {e}")
                    continue

            return jsonify({"response": "❌ Analyse image impossible. Réessaie."})

        except Exception as e:
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # ── 💬 TEXTE ──
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    try:
        return jsonify(handle_chat(user_message, history))
    except Exception as e:
        print("[CHAT ERR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
