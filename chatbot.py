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
# IMAGE TYPES
# ─────────────────────────────
TYPE_PROMPTS = {
    "logo": "minimalist flat vector logo, clean design",
    "icon": "simple app icon, bold shape",
    "illustration": "african digital illustration, colorful, detailed",
    "photo": "photorealistic DSLR quality",
    "pattern": "african textile seamless pattern",
    "banner": "modern wide banner design",
    "avatar": "portrait centered face",
    "poster": "event poster design",
    "general": "high quality digital artwork"
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
# 🔥 DETECTION IMAGE ROBUSTE
# ─────────────────────────────
def detect_image_intent(text):

    lower = text.lower()

    # FORCAGE KEYWORDS (IMPORTANT)
    keywords = [
        "logo", "image", "dessine", "crée", "génère",
        "illustration", "photo", "avatar", "poster", "banner"
    ]

    if any(k in lower for k in keywords):
        return {
            "is_image_request": True,
            "type": "general",
            "visual_prompt": text,
            "confirmation_message": "🎨 Génération de votre image..."
        }

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""
Analyse ce message et réponds UNIQUEMENT en JSON.

Message: {text}

Si image:
{{
  "is_image_request": true,
  "type": "logo|icon|illustration|photo|pattern|banner|avatar|poster|general",
  "visual_prompt": "english prompt for AI image generation",
  "confirmation_message": "Je génère votre image"
}}

Sinon:
{{"is_image_request": false}}
"""
            }],
            temperature=0,
            max_tokens=200
        )

        raw = r.choices[0].message.content
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)
        return data if data.get("is_image_request") else None

    except Exception as e:
        print("[DETECT ERROR]", e)
        return None


# ─────────────────────────────
# 🎨 GENERATION IMAGE
# ─────────────────────────────
def generate_image(prompt, gen_type):

    prefix = TYPE_PROMPTS.get(gen_type, "")
    full_prompt = f"{prefix}, high quality, {prompt}"

    width, height = TYPE_SIZES.get(gen_type, (768, 768))

    # ── TOGETHER AI (FLUX) ──
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

    # ── FALLBACK ──
    try:
        encoded = urllib.parse.quote(full_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"

        img = requests.get(url, timeout=90)

        return base64.b64encode(img.content).decode()

    except Exception as e:
        print("[FALLBACK ERROR]", e)
        return None


# ─────────────────────────────
# 💬 CHAT ROUTE
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
    # 🎙 AUDIO (WHISPER)
    # ─────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                text = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=("audio.wav", f, "audio/wav"),
                    language="fr",
                    response_format="text"
                )

            os.unlink(path)
            user_message = text

        except Exception as e:
            return jsonify({"response": f"❌ Audio error: {str(e)}"})

    # ─────────────────────────────
    # 🖼 IMAGE ANALYSIS
    # ─────────────────────────────
    if has_image and image_base64:
        try:
            r = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }},
                        {"type": "text", "text": "Décris cette image"}
                    ]
                }],
                max_tokens=800
            )

            return jsonify({"response": r.choices[0].message.content})

        except Exception as e:
            return jsonify({"response": str(e)})

    # ─────────────────────────────
    # 🎨 IMAGE DETECTION
    # ─────────────────────────────
    intent = detect_image_intent(user_message)

    # 🔥 FORCE SAFE CHECK
    if intent and intent.get("is_image_request"):

        img = generate_image(
            intent.get("visual_prompt", user_message),
            intent.get("type", "general")
        )

        if img:
            return jsonify({
                "response": intent.get("confirmation_message", "Je génère..."),
                "has_image": True,
                "image_base64": img,
                "image_type": intent.get("type", "general"),
                "visual_prompt": intent.get("visual_prompt", user_message)
            })

        return jsonify({"response": "❌ Erreur génération image"})

    # ─────────────────────────────
    # 💬 CHAT NORMAL (FIX IMPORTANT)
    # ─────────────────────────────
    chat_history.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """
Tu es KoraChat.

IMPORTANT :
- Tu n'es pas un assistant textuel limité.
- Tu peux répondre, analyser images, et générer des visuels.
- Ne dis jamais que tu es limité au texte.
- Réponds toujours en français.
"""
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
