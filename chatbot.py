import os
import base64
import tempfile
import urllib.parse
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from gtts import gTTS

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
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
def detect_language(text: str) -> str:
    wolof_keywords = ["nanga", "mangi", "dafa", "wolof", "jërejëf"]
    if any(w in text.lower() for w in wolof_keywords):
        return "wolof"
    return "french"

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
# TEXT TO SPEECH
# ─────────────────────────────────────────────
def text_to_speech_base64(text: str, lang: str = "fr"):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        tts = gTTS(text=text, lang=lang)
        tts.save(tmp_path)

        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    except Exception as e:
        print("[TTS ERROR]", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

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

    lang = detect_language(user_message)
    system = "Tu es un assistant wolof/français." if lang == "wolof" else "You are a French assistant."

    messages = [{"role": "system", "content": system}]
    messages += history[-10:]
    messages.append({"role": "user", "content": user_message})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=600,
    )

    response_text = r.choices[0].message.content

    result = {"response": response_text}

    # On ne génère l'audio de réponse que si le client le demande
    # (ex: l'utilisateur a envoyé un vocal) pour ne pas surcharger
    # inutilement les requêtes texte classiques si besoin de couper ce comportement.
    if want_audio_response:
        audio_b64 = text_to_speech_base64(response_text, lang="fr" if lang != "wolof" else "fr")
        result["audio_base64"] = audio_b64

    return result

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "pong"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    has_audio = bool(data.get("has_audio"))
    audio_base64 = data.get("audio_base64")
    history = data.get("history", [])

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
