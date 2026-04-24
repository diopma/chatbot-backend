import os
import base64
import tempfile
import urllib.parse
import json
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# ─────────────────────────────
# INIT
# ─────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY manquant")

client = Groq(api_key=GROQ_API_KEY)

chat_history = []

# ─────────────────────────────
# IMAGE STYLES
# ─────────────────────────────
TYPE_PROMPTS = {
    "logo": "minimalist flat vector logo, clean design, centered",
    "icon": "simple app icon, flat design",
    "illustration": "african art illustration, colorful, detailed",
    "photo": "photorealistic DSLR quality",
    "pattern": "african textile seamless pattern",
    "banner": "modern wide banner design",
    "avatar": "profile portrait centered face",
    "poster": "event poster, bold typography",
    "general": "high quality digital art"
}

TYPE_SIZES = {
    "logo": (1024, 1024),
    "icon": (512, 512),
    "illustration": (768, 1024),
    "photo": (768, 768),
    "pattern": (1024, 1024),
    "banner": (1200, 630),
    "avatar": (512, 512),
    "poster": (768, 1024),
    "general": (768, 768),
}

# ─────────────────────────────
# DETECTION IMAGE (ROBUSTE)
# ─────────────────────────────
def detect_image_intent(user_message: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON. No explanation."
                },
                {
                    "role": "user",
                    "content": f"""
Detect if this is an image generation request:

Message: {user_message}

Return JSON ONLY:

If image:
{{
  "is_image_request": true,
  "type": "logo|icon|illustration|photo|pattern|banner|avatar|poster|general",
  "visual_prompt": "english prompt",
  "confirmation_message": "short french message"
}}

Else:
{{"is_image_request": false}}
"""
                }
            ],
            temperature=0.1,
            max_tokens=300
        )

        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        if data.get("is_image_request") is True:
            return data

        return None

    except Exception as e:
        print("[INTENT ERROR]", e)
        return None

# ─────────────────────────────
# IMAGE GENERATION
# ─────────────────────────────
def generate_image(prompt, gen_type):

    prefix = TYPE_PROMPTS.get(gen_type, "")
    full_prompt = f"{prefix}, high quality, {prompt}"
    width, height = TYPE_SIZES.get(gen_type, (768, 768))

    # ── FLUX (Together AI) ──
    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt": full_prompt,
                    "width": min(width, 1024),
                    "height": min(height, 1024),
                    "steps": 4,
                    "n": 1,
                    "response_format": "b64_json",
                },
                timeout=90
            )
            r.raise_for_status()
            return r.json()["data"][0]["b64_json"]

        except Exception as e:
            print("[FLUX ERROR]", e)

    # ── FALLBACK POLLINATIONS ──
    try:
        encoded = urllib.parse.quote(full_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
        img = requests.get(url, timeout=90)
        return base64.b64encode(img.content).decode()

    except Exception as e:
        print("[FALLBACK ERROR]", e)
        return None


# ─────────────────────────────
# DETECT IMAGE MIME TYPE
# ─────────────────────────────
def get_image_media_type(image_base64: str) -> str:
    """Détecte le type MIME depuis les magic bytes de l'image."""
    try:
        header = base64.b64decode(image_base64[:16])
        if header[:4] == b'\x89PNG':
            return "image/png"
        elif header[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif b'WEBP' in header:
            return "image/webp"
        elif header[:4] == b'GIF8':
            return "image/gif"
    except Exception:
        pass
    return "image/jpeg"  # fallback


# ─────────────────────────────
# CHAT ROUTE
# ─────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    user_message = data.get("message", "")
    has_image = data.get("has_image", False)
    image_base64 = data.get("image_base64")
    has_audio = data.get("has_audio", False)
    audio_base64 = data.get("audio_base64")

    # ─────────────────────────────
    # 🎙 AUDIO — TRANSCRIPTION + SUITE NORMALE
    # ─────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=("audio.wav", f, "audio/wav"),
                    language="fr",
                    response_format="text"
                )

            os.unlink(path)

            # Gestion robuste du retour (str ou objet)
            user_message = transcription if isinstance(transcription, str) else transcription.text
            print("[TRANSCRIPTION]", repr(user_message))

            if not user_message or not user_message.strip():
                return jsonify({"response": "❌ Transcription vide, réessaie de parler plus clairement."})

        except Exception as e:
            print("[AUDIO ERROR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ─────────────────────────────
    # 🖼 IMAGE ANALYSIS — FORMAT CORRIGÉ
    # ─────────────────────────────
    if has_image and image_base64:
        try:
            # ✅ Nettoie le préfixe data URI si présent (ex: "data:image/jpeg;base64,...")
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            # ✅ Détecte automatiquement le bon type MIME
            media_type = get_image_media_type(image_base64)

            question = user_message.strip() if user_message.strip() else "Décris cette image en détail en français."

            print(f"[IMAGE ANALYSIS] media_type={media_type}, question={question[:60]}")

            r = client.chat.completions.create(
                # ✅ Modèle vision actuel supporté par Groq (llama-4-scout)
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                # ✅ Format data URI correct avec bon media_type
                                "url": f"data:{media_type};base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }],
                max_tokens=1024
            )

            return jsonify({"response": r.choices[0].message.content})

        except Exception as e:
            print("[IMAGE ANALYSIS ERROR]", e)
            return jsonify({"response": f"❌ Erreur analyse image : {str(e)}"})

    # ─────────────────────────────
    # 🎨 DÉTECTION GÉNÉRATION IMAGE
    # ─────────────────────────────
    if user_message.strip():
        intent = detect_image_intent(user_message)

        if intent:
            img = generate_image(intent["visual_prompt"], intent["type"])

            if img:
                return jsonify({
                    "response": intent["confirmation_message"],
                    "has_image": True,
                    "image_base64": img,
                    "image_type": intent["type"],
                    "visual_prompt": intent["visual_prompt"]
                })

            return jsonify({"response": "❌ Génération d'image échouée."})

    # ─────────────────────────────
    # 💬 CHAT NORMAL
    # ─────────────────────────────
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    chat_history.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es KoraChat, assistant IA africain intelligent."
                },
                *chat_history[-10:]
            ],
            temperature=0.7,
            max_tokens=800
        )

        reply = r.choices[0].message.content
        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)})


# ─────────────────────────────
@app.route("/")
def home():
    return "KoraChat API 🚀"


# ─────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
