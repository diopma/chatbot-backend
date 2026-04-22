import os
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import requests

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Définir GROQ_API_KEY dans Render !")

# Together AI pour la génération d'images (FLUX) — gratuit au départ
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
chat_history = []


@app.route("/")
def home():
    return "Serveur OK 🚀"


# =========================================================
# NOUVELLE ROUTE : GÉNÉRATION D'IMAGE
# =========================================================
@app.route("/generate-image", methods=["POST"])
def generate_image():
    """
    Génère une image à partir d'un prompt textuel.
    Utilise Together AI (FLUX.1-schnell) — très rapide et gratuit.
    Fallback : Pollinations AI (100% gratuit, aucune clé requise).
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Requête invalide"}), 400

    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt manquant"}), 400

    # ── Améliore le prompt avec Groq pour de meilleurs résultats ──
    try:
        enhanced = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un expert en prompt engineering pour la génération d'images. "
                        "Réécris le prompt de l'utilisateur en anglais pour qu'il soit plus détaillé, "
                        "artistique et précis. Maximum 100 mots. Réponds UNIQUEMENT avec le prompt, sans explication."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.8,
        )
        enhanced_prompt = enhanced.choices[0].message.content.strip()
        print(f"[IMAGE GEN] Prompt amélioré: {enhanced_prompt}")
    except Exception:
        enhanced_prompt = prompt  # fallback au prompt original

    # ── Tentative 1 : Together AI (FLUX.1-schnell) ──
    if TOGETHER_API_KEY:
        try:
            resp = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt": enhanced_prompt,
                    "width": 768,
                    "height": 768,
                    "steps": 4,
                    "n": 1,
                    "response_format": "b64_json",
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            b64 = result["data"][0]["b64_json"]
            print("[IMAGE GEN] ✅ Together AI réussi")
            return jsonify({
                "image_base64": b64,
                "enhanced_prompt": enhanced_prompt,
                "source": "flux"
            })
        except Exception as e:
            print(f"[IMAGE GEN] ❌ Together AI échoué: {e}")

    # ── Tentative 2 : Pollinations AI (100% gratuit, aucune clé) ──
    try:
        import urllib.parse
        encoded = urllib.parse.quote(enhanced_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&nologo=true&enhance=true"

        img_resp = requests.get(url, timeout=60)
        img_resp.raise_for_status()

        b64 = base64.b64encode(img_resp.content).decode("utf-8")
        print("[IMAGE GEN] ✅ Pollinations AI réussi")
        return jsonify({
            "image_base64": b64,
            "enhanced_prompt": enhanced_prompt,
            "source": "pollinations"
        })

    except Exception as e:
        print(f"[IMAGE GEN] ❌ Pollinations échoué: {e}")
        return jsonify({"error": "Impossible de générer l'image pour le moment."}), 500


# =========================================================
# ROUTE CHAT (inchangée)
# =========================================================
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    data = request.get_json()
    if not data:
        return jsonify({"error": "Requête invalide"}), 400

    user_message = data.get("message", "")
    has_image    = data.get("has_image", False)
    has_audio    = data.get("has_audio", False)
    image_base64 = data.get("image_base64", None)
    image_type   = data.get("image_type", "image/jpeg")
    audio_base64 = data.get("audio_base64", None)
    audio_ext    = data.get("audio_ext", "wav")
    audio_type   = data.get("audio_type", "audio/wav")

    # ── CAS 1 : AUDIO ──
    if has_audio and audio_base64:
        tmp_path = None
        transcribed_text = None

        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"[AUDIO] Taille reçue: {len(audio_bytes)} bytes, format: {audio_ext}")

            formats_to_try = (
                [(".wav","audio/wav"),(".mp3","audio/mpeg"),(".m4a","audio/m4a")]
                if audio_ext == "wav"
                else [(".m4a","audio/m4a"),(".wav","audio/wav"),(".mp3","audio/mpeg")]
            )

            for ext, mime in formats_to_try:
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    with open(tmp_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-large-v3",
                            file=(f"audio{ext}", audio_file, mime),
                            language="fr",
                            response_format="verbose_json",
                            temperature=0.0,
                            prompt="Transcription d'un message vocal en français.",
                        )

                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        tmp_path = None

                    if hasattr(transcription, "text") and transcription.text.strip():
                        transcribed_text = transcription.text.strip()
                        print(f"[AUDIO] ✅ Transcription: '{transcribed_text}'")
                        break

                except Exception as fmt_err:
                    print(f"[AUDIO] ❌ Format {ext}: {fmt_err}")
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        tmp_path = None
                    continue

            if not transcribed_text:
                return jsonify({"response": "❌ Transcription impossible. Parlez plus fort ou rapprochez-vous du micro."})

            user_message = transcribed_text

        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return jsonify({"response": f"Erreur de transcription : {str(e)}"}), 500

    # ── CAS 2 : IMAGE (analyse) ──
    if has_image and image_base64:
        try:
            prompt_text = user_message or "Analyse et décris cette image en détail."

            vision_messages = [
                {"role": "system", "content": "Tu es un assistant visuel. Réponds toujours en français."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text}
                ]}
            ]

            vision_models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.2-11b-vision-preview",
                "llama-3.2-90b-vision-preview",
            ]

            reply = None
            for model in vision_models:
                try:
                    response = client.chat.completions.create(
                        model=model, messages=vision_messages, temperature=0.7, max_tokens=1024,
                    )
                    reply = response.choices[0].message.content
                    print(f"[IMAGE VISION] ✅ {model}")
                    break
                except Exception as me:
                    print(f"[IMAGE VISION] ❌ {model}: {me}")
                    continue

            if not reply:
                reply = "Je ne parviens pas à analyser cette image pour le moment."

            chat_history.append({"role": "user", "content": f"[Image] {prompt_text}"})
            chat_history.append({"role": "assistant", "content": reply})
            return jsonify({"response": reply})

        except Exception as e:
            return jsonify({"response": f"Erreur d'analyse : {str(e)}"}), 500

    # ── CAS 3 : TEXTE ──
    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    chat_history.append({"role": "user", "content": user_message})

    try:
        recent_history = chat_history[-10:]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Tu es un assistant utile, intelligent et amical. Réponds toujours en français, de manière claire et concise."},
                *recent_history
            ],
            temperature=0.7,
            max_tokens=800,
        )

        reply = response.choices[0].message.content or "Réponse vide 🤖"
        chat_history.append({"role": "assistant", "content": reply})

        if has_audio:
            reply = f"🎙️ *« {user_message} »*\n\n{reply}"

        return jsonify({"response": reply})

    except Exception as e:
        print(f"[TEXT ERROR] {e}")
        return jsonify({"error": "Erreur serveur"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
