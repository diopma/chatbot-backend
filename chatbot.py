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

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
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
    "banner":       (1200, 630),
    "avatar":       (512,  512),
    "poster":       (768,  1024),
    "general":      (768,  768),
}

# ─────────────────────────────────────────────────────────────
# DÉTECTION IMAGE — mots-clés STRICTS (évite les faux positifs)
# ─────────────────────────────────────────────────────────────
IMAGE_TRIGGER_PHRASES = [
    # Français — verbes + objet visuel obligatoire
    "génère une", "génère un", "générer une", "générer un",
    "crée une image", "crée un logo", "crée une illustration", "crée un avatar",
    "créer une image", "créer un logo",
    "dessine", "dessiner",
    "fais moi une image", "fais moi un logo", "fais une image", "fais un logo",
    "je veux une image", "je veux un logo", "je veux une illustration",
    "montre moi une image", "montre une image",
    # Anglais
    "generate an image", "generate a logo", "create an image", "create a logo",
    "draw me", "make me an image", "make a logo",
    "render", "imagine a",
]

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo", "marque", "brand"],
    "icon":         ["icône", "icon", "pictogramme"],
    "illustration": ["illustration", "illustre", "dessine", "art africain"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "tissu", "kente", "wax", "textile"],
    "banner":       ["bannière", "banner", "couverture", "cover"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer", "événement"],
}

def quick_detect_image(message: str) -> bool:
    """Détection stricte pour éviter les faux positifs."""
    msg = message.lower()
    return any(phrase in msg for phrase in IMAGE_TRIGGER_PHRASES)

def quick_detect_type(message: str) -> str:
    msg = message.lower()
    for img_type, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return img_type
    return "general"

# ─────────────────────────────────────────────────────────────
# DÉTECTION IA (seulement si mots-clés stricts trouvés)
# ─────────────────────────────────────────────────────────────
def detect_image_intent(user_message: str) -> dict | None:
    if not quick_detect_image(user_message):
        return None

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # rapide
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No explanation. No markdown."},
                {"role": "user",   "content": (
                    f"Is this an image generation request?\nMessage: {user_message}\n\n"
                    'If yes:  {"is_image_request":true,"type":"logo|icon|illustration|photo|pattern|banner|avatar|poster|general","visual_prompt":"english description","confirmation_message":"message français court"}\n'
                    'If no:   {"is_image_request":false}'
                )}
            ],
            temperature=0.1,
            max_tokens=150,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        if data.get("is_image_request") is True:
            return data
        return None

    except Exception as e:
        print("[INTENT ERROR]", e)
        # Fallback par mots-clés si l'IA échoue
        return {
            "is_image_request":    True,
            "type":                quick_detect_type(user_message),
            "visual_prompt":       user_message,
            "confirmation_message":"🎨 Génération en cours...",
        }

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    prefix      = TYPE_PROMPTS.get(gen_type, "")
    full_prompt = f"{prefix}, high quality, {prompt}"
    width, height = TYPE_SIZES.get(gen_type, (768, 768))

    # ── Together AI (FLUX) ──
    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
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

    # ── Fallback Pollinations ──
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
def get_image_media_type(image_base64: str) -> str:
    try:
        header = base64.b64decode(image_base64[:20])
        if header[:4] == b'\x89PNG':    return "image/png"
        elif header[:2] == b'\xff\xd8': return "image/jpeg"
        elif b'WEBP'   in header:       return "image/webp"
        elif header[:4] == b'GIF8':     return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# ROUTE PRINCIPALE /chat
# ─────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    user_message  = data.get("message", "")
    has_image     = data.get("has_image", False)
    image_base64  = data.get("image_base64")
    has_audio     = data.get("has_audio", False)
    audio_base64  = data.get("audio_base64")

    # ── Historique par session (sans état global) ──
    # Le frontend doit envoyer l'historique dans "history" (liste de messages)
    history = data.get("history", [])

    # ─────────────────────────────
    # 🎙 AUDIO → TRANSCRIPTION
    # ─────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",  # ✅ plus rapide que large-v3
                    file=("audio.wav", f, "audio/wav"),
                    # ✅ pas de language= forcé → auto-détection de la langue
                    response_format="text",
                )
            os.unlink(path)

            user_message = transcription if isinstance(transcription, str) else transcription.text
            print("[TRANSCRIPTION]", repr(user_message))

            if not user_message or not user_message.strip():
                return jsonify({"response": "❌ Audio non reconnu. Parle plus clairement et réessaie."})

            # La transcription continue vers le chat ou la détection image ci-dessous

        except Exception as e:
            print("[AUDIO ERROR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ─────────────────────────────
    # 🖼 ANALYSE IMAGE
    # ─────────────────────────────
    if has_image and image_base64:
        try:
            # Nettoie le préfixe data URI si présent
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            media_type = get_image_media_type(image_base64)
            question   = user_message.strip() if user_message.strip() else "Décris cette image en détail en français."

            print(f"[IMAGE ANALYSIS] media_type={media_type} | question={question[:60]}")

            # ✅ Modèle vision disponible sur Groq en 2025
            r = client.chat.completions.create(
                model="llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                        {"type": "text",      "text": question},
                    ],
                }],
                max_tokens=1024,
            )
            return jsonify({"response": r.choices[0].message.content})

        except Exception as e:
            print("[IMAGE ANALYSIS ERROR]", e)
            # Essai avec modèle alternatif
            try:
                r = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                            {"type": "text",      "text": question},
                        ],
                    }],
                    max_tokens=1024,
                )
                return jsonify({"response": r.choices[0].message.content})
            except Exception as e2:
                print("[IMAGE FALLBACK ERROR]", e2)
                return jsonify({"response": f"❌ Analyse image impossible. Erreur : {str(e)}"})

    # ─────────────────────────────
    # 🎨 GÉNÉRATION IMAGE
    # ─────────────────────────────
    if user_message.strip():
        intent = detect_image_intent(user_message)
        if intent:
            print(f"[IMAGE GEN] type={intent['type']} | prompt={intent['visual_prompt'][:60]}")
            img = generate_image(intent["visual_prompt"], intent["type"])
            if img:
                return jsonify({
                    "response":     intent["confirmation_message"],
                    "has_image":    True,
                    "image_base64": img,
                    "image_type":   intent["type"],
                    "visual_prompt": intent["visual_prompt"],
                })
            return jsonify({"response": "❌ Génération d'image échouée. Réessaie dans quelques secondes."})

    # ─────────────────────────────
    # 💬 CHAT NORMAL
    # ─────────────────────────────
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    # Construit les messages avec historique reçu du frontend
    messages = [
        {
            "role":    "system",
            "content": (
                "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
                "Tu réponds toujours en français sauf si l'utilisateur écrit dans une autre langue. "
                "Tes réponses sont claires, directes et utiles."
            ),
        }
    ]

    # Ajoute l'historique (max 6 derniers échanges pour la rapidité)
    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ajoute le message actuel
    messages.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        reply = r.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        print("[CHAT ERROR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
