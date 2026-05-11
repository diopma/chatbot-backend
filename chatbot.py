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
    "logo":         "minimalist flat vector logo, clean design, centered",
    "icon":         "simple app icon, flat design",
    "illustration": "african art illustration, colorful, detailed",
    "photo":        "photorealistic DSLR quality",
    "pattern":      "african textile seamless pattern",
    "banner":       "modern wide banner design",
    "avatar":       "profile portrait centered face",
    "poster":       "event poster, bold typography",
    "general":      "high quality digital art",
}

TYPE_SIZES = {
    "logo":         (1024, 1024),
    "icon":         (512,  512),
    "illustration": (768,  1024),
    "photo":        (768,  768),
    "pattern":      (1024, 1024),
    "banner":       (1024, 512),
    "avatar":       (512,  512),
    "poster":       (768,  1024),
    "general":      (768,  768),
}

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo"],
    "icon":         ["icône", "icone", "icon"],
    "illustration": ["illustration", "art", "dessin"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
}

# ─────────────────────────────────────────────────────────────
# WOLOF — vocabulaire étendu
# ─────────────────────────────────────────────────────────────
WOLOF_IMAGE_VERBS = [
    "def", "defal", "defe", "deflu",
    "bind", "bindal", "bindaale",
    "yëgël", "yegal", "yëgëlal",
    "seetaan", "seetal", "seete",
    "jëfandikoo", "jëfandiku",
    "am", "amal",
    "teg", "tegal",
    "daldi", "daldi def",
]

WOLOF_VISUAL_NOUNS = [
    "nataal", "nataalu", "nataalyi",
    "logo", "logos",
    "avatar", "avatars",
    "bannière", "affiche",
    "dessin", "motif", "pattern",
    "illustration", "poster", "flyer",
    "liggéey", "liggeeyu",
    "seen nataal", "sama nataal",
]

# Marqueurs wolof pondérés (score élevé = mot très spécifique au wolof)
WOLOF_MARKERS_WEIGHTED = {
    # salutations / formules — très spécifiques
    "nanga def": 3, "mangi fi": 3, "mangi dem": 3,
    "jërejëf": 3, "jërëjëf": 3, "baal ma": 3,
    "yëndul": 3, "waaw waaw": 3, "deedeet": 3,
    "asalaa maalekum": 2, "maalekum salaam": 2,

    # pronoms / particules — très fréquents en wolof
    "mangi": 2, "dama": 2, "dafa": 2, "maa ngi": 2,
    "laa": 2, "naa": 2, "nga": 2, "mu": 2,
    "yow": 2, "moom": 2, "niit": 2,

    # verbes courants
    "dem": 1, "ñëw": 1, "lekk": 1, "dox": 1,
    "fëkk": 1, "nekk": 1, "topp": 1, "wax": 1,
    "xam": 1, "sëdd": 1, "bind": 1, "jëf": 1,
    "tëdd": 1, "daldi": 1, "bëgg": 1, "soxor": 1,

    # mots courants
    "ndax": 1, "ndaxte": 1, "waaye": 1, "wante": 1,
    "mbaa": 1, "ak": 1, "seen": 1, "sama": 1,
    "bi": 1, "yi": 1, "si": 1, "ci": 1, "bu": 1,
    "mbokk": 1, "jabar": 1, "xale": 1, "baay": 1, "yaay": 1,
    "xarit": 1, "doomi": 1, "goor": 1, "jigéen": 1,

    # nombres
    "benn": 1, "ñaar": 1, "ñett": 1, "ñent": 1,
    "juróom": 1, "fukk": 1, "temer": 1,

    # expressions typiques
    "li": 1, "lii": 1, "lool": 1, "dëkk": 1,
    "sunu": 1, "seen": 1, "leen": 1,
}

def detect_language(text: str) -> tuple[str, int]:
    """
    Retourne ('wolof', score), ('french', score), ou ('other', 0).
    Score = confiance de détection.
    """
    t = text.lower()

    # Score wolof
    wolof_score = 0
    for marker, weight in WOLOF_MARKERS_WEIGHTED.items():
        if marker in t:
            wolof_score += weight

    # Score français
    french_markers = {
        "je": 1, "tu": 1, "il": 1, "elle": 1, "nous": 1, "vous": 1,
        "les": 1, "des": 1, "une": 1, "est": 1, "avec": 1,
        "bonjour": 2, "merci": 2, "comment": 1, "pourquoi": 2,
        "parce": 2, "alors": 1, "mais": 1, "donc": 1,
    }
    french_score = sum(w for m, w in french_markers.items() if f" {m} " in f" {t} ")

    if wolof_score >= 2:
        return "wolof", wolof_score
    if wolof_score >= 1 and french_score >= 1:
        return "wolof_french", wolof_score  # mélange = traiter comme wolof
    if french_score >= 2:
        return "french", french_score
    return "other", 0

# ─────────────────────────────────────────────────────────────
# DÉTECTION IMAGE (français + wolof)
# ─────────────────────────────────────────────────────────────
ACTION_VERBS = [
    "génère","générer","genere","generer",
    "crée","créer","cree","creer",
    "dessine","dessiner",
    "fais","faire","fais-moi",
    "montre","montrer",
    "produis","produire",
    "réalise","realise",
    "generate","create","draw","make","render","produce","design","imagine",
] + WOLOF_IMAGE_VERBS

VISUAL_NOUNS = [
    "logo","logos",
    "icône","icones","icon","icons",
    "illustration","illustrations",
    "avatar","avatars",
    "bannière","banniere","banner","banners",
    "affiche","poster","posters","flyer","flyers",
    "motif","pattern","patterns",
    "visuel","visuels",
    "dessin","dessins",
    "portrait","portraits",
] + WOLOF_VISUAL_NOUNS

def _normalize(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a")
        .replace("ù","u").replace("û","u")
        .replace("î","i").replace("ï","i")
        .replace("ô","o").replace("ç","c"))

def detect_image_intent(user_message: str) -> dict | None:
    msg   = _normalize(user_message)
    words = msg.split()
    has_verb = any(_normalize(v) in words or _normalize(v) in msg for v in ACTION_VERBS)
    has_noun = any(n in words for n in VISUAL_NOUNS)
    is_image = (has_verb and has_noun) or has_noun
    if not is_image:
        return None
    gen_type = "general"
    for t, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            gen_type = t
            break
    return {
        "is_image_request":     True,
        "type":                 gen_type,
        "visual_prompt":        user_message,
        "confirmation_message": "🎨 Image générée !",
    }

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    prefix        = TYPE_PROMPTS.get(gen_type, "")
    full_prompt   = f"{prefix}, high quality, {prompt}"
    width, height = TYPE_SIZES.get(gen_type, (768, 768))

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
                    "prompt":          full_prompt,
                    "width":           min(width, 1024),
                    "height":          min(height, 1024),
                    "steps":           4,
                    "n":               1,
                    "response_format": "b64_json",
                },
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["data"][0]["b64_json"]
        except Exception as e:
            print("[FLUX ERROR]", e)

    try:
        encoded = urllib.parse.quote(full_prompt)
        url     = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
        img     = requests.get(url, timeout=90)
        return base64.b64encode(img.content).decode()
    except Exception as e:
        print("[POLLINATIONS ERROR]", e)
        return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION MIME
# ─────────────────────────────────────────────────────────────
def get_image_media_type(b64: str) -> str:
    try:
        h = base64.b64decode(b64[:20])
        if h[:4] == b'\x89PNG':  return "image/png"
        if h[:2] == b'\xff\xd8': return "image/jpeg"
        if b'WEBP' in h:         return "image/webp"
        if h[:4] == b'GIF8':     return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION AUDIO — avec prompt Wolof pour Whisper
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcrit l'audio avec Whisper.
    On tente d'abord sans langue forcée (auto-détection),
    puis avec prompt Wolof si le résultat semble mauvais.
    """
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

    print(f"[AUDIO] format={suffix} taille={len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        with open(path, "rb") as f:
            # Prompt multilingue : aide Whisper à reconnaître
            # le français, le wolof et les termes mixtes
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",          # ← large-v3 (meilleur que turbo pour langues rares)
                file=(f"audio{suffix}", f, mime),
                response_format="text",
                prompt=(
                    "Ce message peut être en français, en wolof, ou un mélange des deux. "
                    "Wolof: nanga def, waaw, deedeet, sama, xam, dafa, mangi, jëf, "
                    "ndax, mbokk, xale, baay, yaay, dem, ñëw, lekk, wax, nekk, topp. "
                    "Termes techniques possibles: logo, image, avatar, créer, générer."
                ),
                # Pas de language= forcé → auto-détection Whisper
            )
    finally:
        os.unlink(path)

    result = transcription if isinstance(transcription, str) else transcription.text
    return (result or "").strip()

# ─────────────────────────────────────────────────────────────
# HANDLE CHAT (texte → image ou réponse)
# ─────────────────────────────────────────────────────────────
def handle_chat(user_message: str, history: list) -> dict:
    # Génération image ?
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
        return {"response": "❌ Génération échouée. Réessaie."}

    # Détecter la langue
    lang, score = detect_language(user_message)
    print(f"[LANG] {lang} (score={score})")

    if lang in ("wolof", "wolof_french"):
        system = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles couramment le wolof et le français, comme un Sénégalais cultivé.

RÈGLES IMPORTANTES :
1. Si l'utilisateur parle wolof → réponds EN WOLOF en priorité, avec du français si nécessaire (code-switching naturel).
2. Si l'utilisateur mélange wolof et français → réponds dans le même mélange naturel.
3. Adapte ton registre : familier avec les jeunes, respectueux avec les aînés.
4. Pour créer une image/logo, dis : "Wax ko ci wolof: 'def ma yenn nataal bu...' walla 'bind ma logo bu...'"
5. N'invente pas de mots wolof — si tu ne sais pas, mélange avec du français.

EXPRESSIONS WOLOF UTILES :
- Salutations: "Nanga def ?", "Mangi fi rekk", "Jërejëf"
- Accord: "Waaw", "Waaw waaw", "Siiw"
- Refus/excuse: "Deedeet", "Baal ma"
- Encouragement: "Yëgël na !", "Baax na !"
- "Je comprends": "Xam naa"
- "C'est bien": "Baax na", "Neex na"
- "Pas de problème": "Amul solo"
- "Allons-y": "Daldi dem"

Exemples de réponses naturelles :
- User: "Nanga def ?" → "Mangi fi rekk, jërejëf ! Yow noo ? Lan la be nelaw ?"
- User: "Dama bëgg xam loolu" → "Waaw, maa ngi wax la ci. [explication]. Xam naa ?"
- User: "Comment on dit merci en wolof ?" → "Ci wolof, 'merci' mooy 'jërejëf'. Neex na bañ !"
"""
    else:
        system = (
            "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
            "Tu réponds en français sauf si l'utilisateur écrit dans une autre langue. "
            "Tu ne prétends JAMAIS ne pas pouvoir créer d'images ou de logos — "
            "si l'utilisateur demande une image, dis-lui d'utiliser des mots comme "
            "'crée un logo', 'génère une image', etc."
        )

    messages = [{"role": "system", "content": system}]

    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.75,
        max_tokens=600,
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

    # ──────────────────────────────
    # 🎙 AUDIO → TRANSCRIPTION → CHAT
    # ──────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)

            if len(audio_bytes) < 1000:
                return jsonify({"response": "❌ Audio trop court. Parle plus longtemps."})

            transcribed = transcribe_audio(audio_bytes)
            print("[TRANSCRIPTION]", repr(transcribed))

            if not transcribed:
                return jsonify({"response": "❌ Audio non reconnu. Parle plus fort ou plus clairement."})

            result = handle_chat(transcribed, history)
            result["transcription"] = transcribed
            return jsonify(result)

        except Exception as e:
            print("[AUDIO ERROR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ──────────────────────────────
    # 🖼 ANALYSE IMAGE
    # ──────────────────────────────
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if not image_base64 or len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})

            media_type = get_image_media_type(image_base64)
            question   = user_message.strip() or "Décris cette image en détail en français."
            print(f"[IMAGE ANALYSIS] {media_type} — {len(image_base64)} chars")

            for model in [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
            ]:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                            {"type": "text", "text": question},
                        ]}],
                        max_tokens=1024,
                    )
                    return jsonify({"response": r.choices[0].message.content})
                except Exception as e:
                    print(f"[IMAGE MODEL ERROR] {model}: {e}")
                    continue

            return jsonify({"response": "❌ Analyse image impossible. Réessaie."})

        except Exception as e:
            print("[IMAGE ANALYSIS ERROR]", e)
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # ──────────────────────────────
    # 💬 TEXTE → CHAT
    # ──────────────────────────────
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    try:
        return jsonify(handle_chat(user_message, history))
    except Exception as e:
        print("[CHAT ERROR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
