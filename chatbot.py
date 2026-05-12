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
    "logo":         "minimalist professional flat vector logo, clean sharp design, centered, white background, high contrast, bold colors",
    "icon":         "simple modern app icon, flat design, bold colors, clean lines, centered",
    "illustration": "vibrant african art illustration, rich colors, detailed, professional digital art",
    "photo":        "photorealistic DSLR quality photo, sharp focus, professional lighting, 8k resolution",
    "pattern":      "beautiful african textile seamless pattern, vibrant kente colors, detailed weave",
    "banner":       "modern professional wide banner design, bold typography, vibrant colors",
    "avatar":       "professional portrait, centered face, studio lighting, sharp details",
    "poster":       "eye-catching event poster, bold typography, vibrant colors, professional design",
    "general":      "high quality professional digital art, vibrant colors, sharp details, 4k",
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
    "illustration": ["illustration", "art", "dessin", "nataal"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
}

# ─────────────────────────────────────────────────────────────
# WOLOF — vocabulaire image
# ─────────────────────────────────────────────────────────────
WOLOF_IMAGE_VERBS = [
    "def", "defal", "defe", "deflu",
    "bind", "bindal", "bindaale",
    "yëgël", "yegal", "yëgëlal",
    "seetaan", "seetal",
    "jëfandikoo", "jëfandiku",
    "am", "amal", "teg", "tegal",
    "daldi def", "daldi bind",
    "yokk", "yokkal",
]

WOLOF_VISUAL_NOUNS = [
    "nataal", "nataalu", "nataalyi",
    "logo", "logos", "avatar", "bannière",
    "affiche", "dessin", "motif", "pattern",
    "illustration", "poster", "flyer",
    "liggéey", "seen nataal", "sama nataal",
]

# ─────────────────────────────────────────────────────────────
# PEULH / TOUCOULEUR — vocabulaire image
# ─────────────────────────────────────────────────────────────
FULA_IMAGE_VERBS = [
    "weel", "weelaa",        # faire / créer
    "winduu", "windu",       # écrire / dessiner
    "hollu", "holl",         # montrer
    "waɗ", "waɗaa",         # faire
    "selminde", "selmin",    # préparer / faire
    "addaa", "adda",         # apporter / créer
]

FULA_VISUAL_NOUNS = [
    "natal", "natali",       # image / photo (peulh)
    "logo", "avatar",
    "bannière", "affiche",
    "dessin", "motif",
    "illustration", "poster",
    "seedantaari",           # preuve / image
]

# ─────────────────────────────────────────────────────────────
# MARQUEURS DE LANGUE pondérés
# ─────────────────────────────────────────────────────────────
WOLOF_MARKERS_WEIGHTED = {
    "nanga def": 3, "mangi fi": 3, "mangi dem": 3,
    "jërejëf": 3, "jërëjëf": 3, "baal ma": 3,
    "waaw waaw": 3, "deedeet": 3, "maa ngi": 2,
    "asalaa maalekum": 2, "mangi": 2, "dama": 2,
    "dafa": 2, "laa": 2, "naa": 2, "nga": 2,
    "yow": 2, "moom": 2, "niit": 2, "waaw": 2,
    "dem": 1, "ñëw": 1, "lekk": 1, "dox": 1,
    "fëkk": 1, "nekk": 1, "topp": 1, "wax": 1,
    "xam": 1, "sëdd": 1, "bind": 1, "jëf": 1,
    "tëdd": 1, "bëgg": 1, "ndax": 1, "waaye": 1,
    "mbaa": 1, "ak": 1, "seen": 1, "sama": 1,
    "bi": 1, "yi": 1, "si": 1, "ci": 1, "bu": 1,
    "mbokk": 1, "xale": 1, "baay": 1, "yaay": 1,
    "xarit": 1, "goor": 1, "jigéen": 1,
    "benn": 1, "ñaar": 1, "ñett": 1, "juróom": 1,
    "lool": 1, "dëkk": 1, "sunu": 1, "leen": 1,
    "nataal": 2, "def ma": 2, "bind ma": 2,
}

FULA_MARKERS_WEIGHTED = {
    # salutations / formules
    "jam tan": 3, "jam waali": 3, "a jaraama": 3,
    "on jaraama": 3, "useko": 2, "barakaa": 2,
    "mi yiɗi": 3, "mi faamaaki": 3,
    # pronoms
    "mi": 2, "ɓe": 2, "ko": 2, "ɗum": 2,
    "aan": 2, "min": 2, "on": 1,
    # verbes courants
    "waɗi": 2, "waɗaa": 2, "yahi": 1, "wari": 1,
    "nani": 1, "faami": 1, "hollu": 2, "windu": 2,
    "haal": 1, "yiɗi": 2, "heli": 1,
    # mots courants
    "ko": 1, "ɗoo": 1, "ɗon": 1, "nde": 1,
    "fulo": 1, "pullo": 2, "fulɓe": 2,
    "toucouleur": 2, "haalpulaar": 3, "haal pulaar": 3,
    "pulaar": 3, "peulh": 2,
    # nombres
    "go'o": 1, "ɗiɗi": 1, "tati": 1, "nayi": 1,
    # mots typiques
    "natal": 2, "waɗ ma": 2, "hollu mi": 2,
}

def detect_language(text: str) -> tuple:
    t = text.lower()

    wolof_score = sum(w for m, w in WOLOF_MARKERS_WEIGHTED.items() if m in t)
    fula_score  = sum(w for m, w in FULA_MARKERS_WEIGHTED.items()  if m in t)

    french_markers = {
        "je": 1, "tu": 1, "il": 1, "elle": 1, "nous": 1,
        "les": 1, "des": 1, "une": 1, "est": 1, "avec": 1,
        "bonjour": 2, "merci": 2, "comment": 1, "pourquoi": 2,
        "mais": 1, "donc": 1, "alors": 1,
    }
    french_score = sum(w for m, w in french_markers.items() if f" {m} " in f" {t} ")

    best = max(wolof_score, fula_score, french_score)
    if best == 0:
        return "other", 0
    if wolof_score == best and wolof_score >= 2:
        if french_score >= 2:
            return "wolof_french", wolof_score
        return "wolof", wolof_score
    if fula_score == best and fula_score >= 2:
        if french_score >= 2:
            return "fula_french", fula_score
        return "fula", fula_score
    if french_score >= 2:
        return "french", french_score
    return "other", 0

# ─────────────────────────────────────────────────────────────
# DÉTECTION IMAGE — français + wolof + peulh
# ─────────────────────────────────────────────────────────────
ACTION_VERBS = [
    # français / anglais
    "génère","générer","genere","generer",
    "crée","créer","cree","creer",
    "dessine","dessiner","fais","faire","fais-moi",
    "montre","montrer","produis","produire",
    "réalise","realise","generate","create",
    "draw","make","render","produce","design","imagine",
] + WOLOF_IMAGE_VERBS + FULA_IMAGE_VERBS

VISUAL_NOUNS = [
    "logo","logos","icône","icones","icon","icons",
    "illustration","illustrations","avatar","avatars",
    "bannière","banniere","banner","banners",
    "affiche","poster","posters","flyer","flyers",
    "motif","pattern","patterns","visuel","visuels",
    "dessin","dessins","portrait","portraits",
] + WOLOF_VISUAL_NOUNS + FULA_VISUAL_NOUNS

def _normalize(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a")
        .replace("ù","u").replace("û","u")
        .replace("î","i").replace("ï","i")
        .replace("ô","o").replace("ç","c")
        .replace("ɓ","b").replace("ɗ","d")
        .replace("ŋ","n").replace("ɲ","n"))

def detect_image_intent(user_message: str) -> dict | None:
    msg   = _normalize(user_message)
    words = msg.split()
    has_verb = any(_normalize(v) in words or _normalize(v) in msg for v in ACTION_VERBS)
    has_noun = any(n in words or n in msg for n in VISUAL_NOUNS)
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
# TRADUCTION PROMPT IMAGE vers l'anglais via LLM
# Pour améliorer la qualité des images générées
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str) -> str:
    """Traduit le prompt en anglais descriptif pour FLUX."""
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an image prompt translator. "
                        "Translate the user's request into a detailed English image generation prompt. "
                        "Be descriptive, add quality keywords. "
                        "Return ONLY the English prompt, nothing else. "
                        "Example: 'logo pour restaurant africain' → "
                        "'professional minimalist logo for an african restaurant, "
                        "warm colors, modern typography, clean design'"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
        )
        translated = r.choices[0].message.content.strip()
        print(f"[PROMPT TRANSLATED] {translated}")
        return translated
    except Exception as e:
        print(f"[TRANSLATE ERROR] {e}")
        return prompt  # fallback sur le prompt original

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE — qualité améliorée
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    # Traduire le prompt en anglais pour meilleure qualité
    english_prompt = translate_prompt_to_english(prompt)

    prefix        = TYPE_PROMPTS.get(gen_type, "")
    full_prompt   = f"{prefix}, {english_prompt}, masterpiece, best quality, ultra detailed"
    width, height = TYPE_SIZES.get(gen_type, (1024, 1024))

    print(f"[IMAGE GEN] type={gen_type} prompt={full_prompt[:80]}...")

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

    # Fallback Pollinations — avec paramètres qualité
    try:
        encoded = urllib.parse.quote(full_prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&nologo=true&enhance=true&model=flux"
        )
        img = requests.get(url, timeout=90)
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
# TRANSCRIPTION AUDIO
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
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
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio{suffix}", f, mime),
                response_format="text",
                prompt=(
                    "Ce message peut être en français, wolof, pulaar/peulh/toucouleur, "
                    "ou un mélange. "
                    "Wolof: nanga def, mangi fi, jërejëf, waaw, deedeet, dama, dafa, "
                    "xam, dem, ñëw, lekk, wax, nekk, bi, yi, ci, ak, sama, bëgg, "
                    "nataal, def ma, bind ma, logo. "
                    "Pulaar/Peulh: jam tan, a jaraama, mi yiɗi, waɗi, hollu, windu, "
                    "natal, ko, ɗum, aan, mi, fulɓe, haalpulaar, pulaar. "
                    "Termes tech: logo, image, avatar, créer, générer, dessiner."
                ),
            )
    finally:
        os.unlink(path)

    result = transcription if isinstance(transcription, str) else transcription.text
    return (result or "").strip()

# ─────────────────────────────────────────────────────────────
# HANDLE CHAT
# ─────────────────────────────────────────────────────────────
def handle_chat(user_message: str, history: list) -> dict:
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

    lang, score = detect_language(user_message)
    print(f"[LANG] {lang} (score={score})")

    if lang in ("wolof", "wolof_french"):
        system = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles couramment wolof et français, comme un Sénégalais cultivé de Dakar.

RÈGLES :
1. Réponds EN WOLOF en priorité, avec du français si nécessaire (code-switching naturel).
2. Si mélange wolof-français → réponds dans le même style naturel.
3. Adapte : familier avec les jeunes, respectueux avec les aînés.
4. Pour créer une image/logo → dis : "Wax : 'def ma logo bu...' walla 'bind ma nataal bu...'"

EXPRESSIONS UTILES :
- Salut: "Nanga def ?", "Mangi fi rekk, yow ?"
- Merci: "Jërejëf", "Jërejëf lool"
- D'accord: "Waaw", "Siiw", "Baax na"
- Excuse: "Baal ma", "Baalma"
- Bien: "Neex na", "Baax na", "Yëgël na"
- Pas de problème: "Amul solo"
- Allons-y: "Daldi dem", "Daldi"
- Je comprends: "Xam naa", "Faamaak"

EXEMPLES :
- "Nanga def ?" → "Mangi fi rekk, jërejëf ! Yow noo ? Lan la be nelaw ?"
- "Dama bëgg xam loolu" → "Waaw, maa ngi wax la ci. [explication]. Xam naa ?"
- "Comment dire merci ?" → "Ci wolof, merci mooy 'jërejëf'. Neex na bañ !"
"""
    elif lang in ("fula", "fula_french"):
        system = """Tu es Yelen AI, un assistant IA sénégalais intelligent et chaleureux.
Tu parles le pulaar/peulh/toucouleur et le français, comme un Haalpulaar du Sénégal.

RÈGLES :
1. Réponds EN PULAAR en priorité, avec du français si nécessaire.
2. Si mélange pulaar-français → réponds dans le même style naturel.
3. Adapte ton registre selon le contexte.
4. Pour créer une image/logo → dis : "Haal : 'waɗ mi natal...' walla 'hollu mi logo...'"

EXPRESSIONS PULAAR UTILES :
- Salut: "Jam tan ?", "A jaraama", "On jaraama"
- Merci: "A jaraama", "Barakaa"
- D'accord: "Eey", "Ɗum noon"
- Excuse: "Yaafo", "Yaafo mi"
- Bien: "Ɗum baɗi", "Baaw ɗum"
- Je comprends: "Mi faami", "Faami mi"
- Pas de problème: "Alaa ko weli"
- Comment vas-tu: "No mbadaa ?"

EXEMPLES :
- "Jam tan ?" → "Jam tan, a jaraama ! Aan noy mbadaa ?"
- "Mi yiɗi faami ɗum" → "Eey, mi haalat ma. [explication]. A faami ?"
- "Comment dire merci ?" → "E pulaar, merci ko 'a jaraama'. Ɗum baɗi !"
"""
    else:
        system = (
            "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
            "Tu réponds en français sauf si l'utilisateur écrit dans une autre langue. "
            "Tu ne prétends JAMAIS ne pas pouvoir créer d'images — "
            "si demandé, dis d'utiliser : 'crée un logo', 'génère une image'."
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

    # 🎙 AUDIO
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            if len(audio_bytes) < 1000:
                return jsonify({"response": "❌ Audio trop court. Parle plus longtemps."})
            transcribed = transcribe_audio(audio_bytes)
            print("[TRANSCRIPTION]", repr(transcribed))
            if not transcribed:
                return jsonify({"response": "❌ Audio non reconnu. Parle plus fort."})
            result = handle_chat(transcribed, history)
            result["transcription"] = transcribed
            return jsonify(result)
        except Exception as e:
            print("[AUDIO ERROR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # 🖼 ANALYSE IMAGE
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if not image_base64 or len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})
            media_type = get_image_media_type(image_base64)
            question   = user_message.strip() or "Décris cette image en détail en français."
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
            return jsonify({"response": "❌ Analyse image impossible."})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # 💬 TEXTE
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
