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
    "illustration": ["illustration", "art", "dessin", "nataal bu"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
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
    # Wolof
    "nataal", "nataalu", "nataalyi",   # image / photo
    "sama nataal", "seen nataal",
    "liggéey bu nataal",
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
    # Wolof
    "def", "defal",       # faire / créer
    "bind", "bindal",     # dessiner / écrire
    "yëgël", "yegal",     # montrer
    "def ma", "bind ma",  # fais-moi / dessine-moi
    "yokk", "yokkal",     # ajouter / faire
    "am", "amal",         # avoir / produire
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
    # Score 3 = très spécifique au wolof
    "nanga def": 3, "mangi fi": 3, "jërejëf": 3, "jërëjëf": 3,
    "baal ma": 3, "deedeet": 3, "maa ngi": 3, "waaw waaw": 3,
    "def ma": 3, "bind ma": 3, "nataal": 3, "nataalu": 3,
    # Score 2 = fréquent en wolof
    "mangi": 2, "dama": 2, "dafa": 2, "waaw": 2, "yow": 2,
    "moom": 2, "laa": 2, "naa": 2, "nga": 2, "niit": 2,
    "xam": 2, "bëgg": 2, "nekk": 2,
    # Score 1 = courant
    "dem": 1, "ñëw": 1, "lekk": 1, "dox": 1, "fëkk": 1,
    "topp": 1, "wax": 1, "sëdd": 1, "bind": 1, "jëf": 1,
    "tëdd": 1, "ndax": 1, "waaye": 1, "mbaa": 1,
    "ak": 1, "sama": 1, "seen": 1, "ci": 1, "bi": 1,
    "yi": 1, "bu": 1, "mbokk": 1, "xale": 1, "baay": 1,
    "yaay": 1, "xarit": 1, "goor": 1, "jigéen": 1,
    "benn": 1, "ñaar": 1, "ñett": 1, "juróom": 1,
    "lool": 1, "sunu": 1, "leen": 1, "def": 1,
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
WOLOF_SYSTEM = """Tu es Yelen AI, un assistant IA sénégalais intelligent et chaleureux.
Tu parles wolof et français couramment, comme un Dakarois éduqué.

RÈGLES STRICTES :
1. Si l'utilisateur parle wolof → réponds EN WOLOF avec du français si nécessaire.
2. Si mélange wolof-français → réponds dans le même mélange naturel (code-switching).
3. Ne force PAS le wolof si la question est en français pur → réponds en français.
4. Sois concis, naturel, chaleureux.
5. N'invente JAMAIS de mots wolof — si tu ne sais pas, dis-le en français.
6. Pour générer une image en wolof : "Wax : 'def ma logo' walla 'bind ma nataal'"

VOCABULAIRE WOLOF DE BASE (utilise-le naturellement) :
- Salut: "Nanga def ?", "Mangi fi rekk"
- Merci: "Jërejëf", "Jërejëf lool"
- Oui: "Waaw", "Waaw waaw"
- Non: "Deedeet"
- Bien: "Baax na", "Neex na"
- D'accord: "Siiw", "Waaw baax na"
- Excuse: "Baal ma"
- Je comprends: "Xam naa", "Faamaak"
- Pas de problème: "Amul solo"
- C'est bon: "Baax na", "Dafa baax"
- Je veux: "Dama bëgg"
- Comment: "Naka"

EXEMPLES DE RÉPONSES NATURELLES :
- "Nanga def ?" → "Mangi fi rekk, jërejëf ! Yow noo ?"
- "Dama bëgg xam..." → "Waaw, maa ngi wax la. [réponse]. Xam naa ?"
- "Logo bi neex na !" → "Jërejëf lool ! Dafa baax moom ?"
- "Def ma yenn logo" → "Waaw ! Wax ma soo bëgg : magasin, restaurant, walla lan ?"
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
