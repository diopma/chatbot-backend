import os
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB pour WAV non compressé

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Définir GROQ_API_KEY dans Render !")

client = Groq(api_key=GROQ_API_KEY)
chat_history = []


@app.route("/")
def home():
    return "Serveur OK 🚀"


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
    audio_ext    = data.get("audio_ext", "wav")   # "wav" ou "m4a" envoyé par le frontend
    audio_type   = data.get("audio_type", "audio/wav")

    # =========================================================
    # CAS 1 : AUDIO → transcription Whisper
    # =========================================================
    if has_audio and audio_base64:
        tmp_path = None
        transcribed_text = None

        try:
            audio_bytes = base64.b64decode(audio_base64)
            print(f"[AUDIO] Taille reçue: {len(audio_bytes)} bytes, format: {audio_ext}")

            # Détermine les formats à essayer selon ce qu'a envoyé le frontend
            if audio_ext == "wav":
                formats_to_try = [
                    (".wav", "audio/wav"),
                    (".mp3", "audio/mpeg"),
                    (".m4a", "audio/m4a"),
                ]
            else:
                formats_to_try = [
                    (".m4a", "audio/m4a"),
                    (".wav", "audio/wav"),
                    (".mp3", "audio/mpeg"),
                ]

            for ext, mime in formats_to_try:
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name

                    print(f"[AUDIO] Essai format {ext}...")

                    with open(tmp_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-large-v3",        # modèle le plus précis
                            file=(f"audio{ext}", audio_file, mime),
                            language="fr",
                            response_format="verbose_json",
                            temperature=0.0,                 # déterministe = plus précis
                            prompt="Transcription d'un message vocal en français. Inclure tous les mots, même courts.",
                        )

                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        tmp_path = None

                    if hasattr(transcription, "text") and transcription.text.strip():
                        transcribed_text = transcription.text.strip()
                        print(f"[AUDIO] ✅ Transcription réussie ({ext}): '{transcribed_text}'")
                        break
                    else:
                        print(f"[AUDIO] ⚠️ Transcription vide avec {ext}")

                except Exception as fmt_err:
                    print(f"[AUDIO] ❌ Format {ext} échoué: {fmt_err}")
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        tmp_path = None
                    continue

            if not transcribed_text:
                return jsonify({
                    "response": "❌ Je n'ai pas pu transcrire votre message. Essayez de parler plus fort ou plus près du micro."
                })

            # Utilise le texte transcrit pour générer une réponse
            user_message = transcribed_text

        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return jsonify({"response": f"Erreur de transcription : {str(e)}"}), 500

    # =========================================================
    # CAS 2 : IMAGE → modèle vision
    # =========================================================
    if has_image and image_base64:
        try:
            prompt_text = user_message if user_message else "Analyse et décris cette image en détail."

            vision_messages = [
                {
                    "role": "system",
                    "content": "Tu es un assistant visuel. Analyse les images avec précision et réponds toujours en français."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_type};base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
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
                        model=model,
                        messages=vision_messages,
                        temperature=0.7,
                        max_tokens=1024,
                    )
                    reply = response.choices[0].message.content
                    print(f"[IMAGE] ✅ Modèle utilisé: {model}")
                    break
                except Exception as model_err:
                    print(f"[IMAGE] ❌ {model}: {model_err}")
                    continue

            if not reply:
                reply = "Je ne parviens pas à analyser cette image pour le moment."

            chat_history.append({"role": "user", "content": f"[Image] {prompt_text}"})
            chat_history.append({"role": "assistant", "content": reply})
            return jsonify({"response": reply})

        except Exception as e:
            print(f"[IMAGE ERROR] {e}")
            return jsonify({"response": f"Erreur d'analyse d'image : {str(e)}"}), 500

    # =========================================================
    # CAS 3 : TEXTE (ou texte issu de transcription audio)
    # =========================================================
    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    chat_history.append({"role": "user", "content": user_message})

    try:
        recent_history = chat_history[-10:]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # modèle plus puissant pour meilleure réponse
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant utile, intelligent et amical. Réponds toujours en français, de manière claire et concise."
                },
                *recent_history
            ],
            temperature=0.7,
            max_tokens=800,
        )

        reply = response.choices[0].message.content or "Réponse vide 🤖"
        chat_history.append({"role": "assistant", "content": reply})

        # Ajoute la transcription dans la réponse audio pour confirmation
        if has_audio:
            reply = f"🎙️ *« {user_message} »*\n\n{reply}"

        return jsonify({"response": reply})

    except Exception as e:
        print(f"[TEXT ERROR] {e}")
        return jsonify({"error": "Erreur serveur"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
