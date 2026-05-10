import os
import base64
import tempfile
import urllib.parse
import json
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
    "banner":       (1024, 512),   # réduit : 1200x630 rejeté par FLUX free
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
# OPTIMISATION : détection image SANS appel LLM supplémentaire
# On utilise uniquement les mots-clés (rapide, 0 latence)
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
]

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
    # NB : "image" et "photo" retirés car trop ambigus (ex: "analyse cette image")
]

def _normalize(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a")
        .replace("ù","u").replace("û","u")
        .replace("î","i").replace("ï","i")
        .replace("ô","o").replace("ç","c"))

def detect_image_intent(user_message: str) -> dict | None:
    """
    Détection rapide par mots-clés uniquement.
    Supprime l'appel LLM intermédiaire (économise ~500ms par requête).
    """
    msg   = _normalize(user_message)
    words = msg.split()

    has_verb  = any(_normalize(v) in words or _normalize(v) in msg for v in ACTION_VERBS)
    has_noun  = any(n in words for n in VISUAL_NOUNS)

    # Verbe + nom visuel, OU nom visuel seul (ex: "logo pour mon resto")
    is_image = (has_verb and has_noun) or has_noun

    if not is_image:
        return None

    # Détecter le type
    gen_type = "general"
    for t, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            gen_type = t
            break

    # Construire un prompt anglais minimal
    visual_prompt = user_message  # Whisper/FLUX gèrent le français

    return {
        "is_image_request":     True,
        "type":                 gen_type,
        "visual_prompt":        visual_prompt,
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

    # Fallback Pollinations
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
        if h[:4] == b'\x89PNG':    return "image/png"
        if h[:2] == b'\xff\xd8':   return "image/jpeg"
        if b'WEBP' in h:           return "image/webp"
        if h[:4] == b'GIF8':       return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# KEEP-ALIVE : empêche le cold start sur Render Free
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
    # 🎙 AUDIO → TRANSCRIPTION
    # ──────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)

            # Détection format par magic bytes
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

            print(f"[AUDIO] format détecté: {suffix} ({len(audio_bytes)} bytes)")

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=(f"audio{suffix}", f, mime),
                    response_format="text",
                )
            os.unlink(path)

            user_message = transcription if isinstance(transcription, str) else transcription.text
            print("[TRANSCRIPTION]", repr(user_message))

            if not user_message or not user_message.strip():
                return jsonify({"response": "❌ Audio non reconnu. Parle plus fort ou plus clairement."})

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
    # 🎨 GÉNÉRATION IMAGE
    # ──────────────────────────────
    if user_message.strip():
        intent = detect_image_intent(user_message)
        if intent:
            img = generate_image(intent["visual_prompt"], intent["type"])
            if img:
                return jsonify({
                    "response":      intent["confirmation_message"],
                    "has_image":     True,
                    "image_base64":  img,
                    "image_type":    intent["type"],
                    "visual_prompt": intent["visual_prompt"],
                })
            return jsonify({"response": "❌ Génération échouée. Réessaie."})

    # ──────────────────────────────
    # 💬 CHAT NORMAL
    # ──────────────────────────────
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    messages = [{
        "role":    "system",
        "content": (
            "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
            "Tu réponds en français sauf si l'utilisateur écrit dans une autre langue. "
            "Tu ne prétends JAMAIS ne pas pouvoir créer d'images ou de logos — "
            "si l'utilisateur demande une image, dis-lui d'utiliser des mots comme "
            "'crée un logo', 'génère une image', etc."
        ),
    }]

    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return jsonify({"response": r.choices[0].message.content})
    except Exception as e:
        print("[CHAT ERROR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
