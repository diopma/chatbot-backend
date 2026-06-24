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
# LANGUAGE / SYSTEM PROMPT
# ─────────────────────────────────────────────
# Plutôt qu'une détection par mots-clés (peu fiable : la plupart des phrases
# en wolof ne contiennent aucun des mots-clés type "nanga"/"mangi"/...), on
# laisse le LLM identifier lui-même la langue du message et y répondre.
# Llama 3.3 a une connaissance limitée du wolof (langue peu présente dans
# les corpus d'entraînement) mais fait un effort correct en best-effort.
SYSTEM_PROMPT = (
    "Tu es Yelen AI, un assistant qui parle français et wolof.\n"
    "Détecte automatiquement la langue du message de l'utilisateur "
    "(français ou wolof) et réponds TOUJOURS dans cette même langue.\n"
    "Si l'utilisateur écrit en wolof, fais de ton mieux pour répondre "
    "entièrement en wolof, même si ta maîtrise du wolof est imparfaite : "
    "ne bascule pas en français sauf si l'utilisateur te le demande "
    "explicitement ou s'il mélange lui-même les deux langues.\n"
    "Si l'utilisateur écrit en français, réponds en français."
)

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
# IMAGE VISION (analyse d'une image envoyée par l'utilisateur)
# ─────────────────────────────────────────────
def _detect_image_mime(raw_bytes: bytes) -> str:
    """Détecte le type MIME réel à partir des premiers octets (signature de fichier)."""
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw_bytes.startswith(b"GIF87a") or raw_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if raw_bytes.startswith(b"RIFF") and raw_bytes[8:12] == b"WEBP":
        return "image/webp"
    # Par défaut : jpeg (format le plus courant depuis les galeries mobiles)
    return "image/jpeg"


def analyze_image_base64(image_base64: str, question: str):
    """
    Envoie l'image (base64) + une question à un modèle vision via Groq
    (Llama 4 Scout) et retourne la réponse texte du modèle, ou None en
    cas d'échec.
    """
    try:
        raw_bytes = base64.b64decode(image_base64)
        mime = _detect_image_mime(raw_bytes)
        data_url = f"data:{mime};base64,{image_base64}"
        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
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
# DOCUMENT (extraction + analyse de PDF)
# ─────────────────────────────────────────────
MAX_DOC_CHARS = 15000  # limite de texte envoyée au LLM pour rester dans le contexte

def extract_pdf_text(pdf_base64: str):
    """
    Décode un PDF en base64 et en extrait le texte (toutes pages, tronqué
    si trop long). Retourne (texte, erreur) ; texte est None si l'extraction
    échoue (PDF scanné sans texte, fichier corrompu, etc.).
    """
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


def analyze_document(doc_text: str, question: str):
    """Envoie le texte extrait du document + la question de l'utilisateur au LLM."""
    try:
        prompt = (
            "Voici le contenu d'un document fourni par l'utilisateur :\n\n"
            f"---\n{doc_text}\n---\n\n"
            f"Question de l'utilisateur : {question}"
        )
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
    """Wrapper synchrone pour edge_tts (lib asyncio)."""
    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)
    asyncio.run(_run())


def text_to_speech_base64(text: str, lang: str = "fr", max_retries: int = 2):
    """
    Génère un mp3 de la réponse.

    Priorité 1 : edge-tts — s'appuie sur l'infrastructure officielle de
    synthèse vocale de Microsoft Edge (Read Aloud), beaucoup plus stable
    en environnement serveur/cloud que gTTS.

    Priorité 2 (fallback) : gTTS — endpoint non-officiel de Google
    Translate ; peut renvoyer 403/429 selon l'IP sortante de l'hébergeur
    (observé sur certaines IP partagées de type Render).

    Retourne (audio_base64, message_erreur). message_erreur est None en
    cas de succès, sinon contient le détail des deux échecs pour debug
    direct dans les logs serveur / réponse JSON.
    """
    voice = "fr-FR-DeniseNeural"
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
    for attempt in range(max_retries + 1):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            tts = gTTS(text=text, lang=lang)
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
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
        audio_b64, tts_error = text_to_speech_base64(response_text)
        result["audio_base64"] = audio_b64
        if tts_error:
            # Visible dans les logs Render ET dans la réponse, pour debug rapide.
            print("[TTS] échec définitif après retries :", tts_error)
            result["tts_error"] = tts_error

    return result

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "pong"

@app.route("/tts", methods=["POST"])
def tts():
    """
    Génère l'audio d'un texte à la demande (bouton "écouter" sur un message
    bot déjà affiché). Le texte est fourni par le client — pas besoin de
    repasser par le LLM, on synthétise directement.
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "texte manquant"}), 400

    if len(text) > 4000:
        text = text[:4000]

    audio_b64, tts_error = text_to_speech_base64(text)

    if not audio_b64:
        return jsonify({"error": tts_error or "échec de la synthèse vocale"}), 502

    return jsonify({"audio_base64": audio_b64})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    has_audio = bool(data.get("has_audio"))
    audio_base64 = data.get("audio_base64")
    has_image = bool(data.get("has_image"))
    image_base64 = data.get("image_base64")
    has_document = bool(data.get("has_document"))
    document_base64 = data.get("document_base64")
    history = data.get("history", [])

    # ── Cas document : extraction texte + analyse, pas besoin de passer par handle_chat ──
    if has_document:
        if not document_base64:
            return jsonify({"error": "document manquant"}), 400

        doc_text, doc_error = extract_pdf_text(document_base64)

        if not doc_text:
            return jsonify({
                "error": doc_error or "Impossible de lire ce document",
                "response": f"❌ {doc_error or 'Je n’ai pas pu lire ce document.'}",
            }), 200

        question = (data.get("message") or "Résume ce document en français.").strip()
        response_text = analyze_document(doc_text, question)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser le document",
                "response": "❌ Je n'ai pas réussi à analyser ce document, réessaie.",
            }), 200

        return jsonify({"response": response_text})

    # ── Cas image : analyse vision directe, pas besoin de passer par handle_chat ──
    if has_image:
        if not image_base64:
            return jsonify({"error": "image manquante"}), 400

        question = (data.get("message") or "Décris cette image en détail en français.").strip()
        response_text = analyze_image_base64(image_base64, question)

        if not response_text:
            return jsonify({
                "error": "Impossible d'analyser l'image",
                "response": "❌ Je n'ai pas réussi à analyser cette image, réessaie.",
            }), 200

        return jsonify({"response": response_text})

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
