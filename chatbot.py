import os
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# Augmente la limite de taille des requêtes à 20MB (pour les images/audio en base64)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Clé Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Définir GROQ_API_KEY dans Render !")

client = Groq(api_key=GROQ_API_KEY)

# Historique global (par utilisateur idéalement, mais simplifié ici)
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

    # =========================================================
    # CAS 1 : MESSAGE AUDIO → transcription Whisper puis réponse
    # =========================================================
    if has_audio and audio_base64:
        try:
            # Décode le base64 en fichier temporaire .m4a
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Transcription avec Whisper (via Groq)
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=("audio.m4a", audio_file, "audio/m4a"),
                    language="fr",
                )
            os.unlink(tmp_path)  # supprime le fichier temporaire

            transcribed_text = transcription.text.strip()
            if not transcribed_text:
                return jsonify({"response": "Je n'ai pas pu entendre votre message vocal."})

            # Utilise le texte transcrit comme message utilisateur
            user_message = transcribed_text
            print(f"[AUDIO] Transcription : {transcribed_text}")

        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
            return jsonify({"response": f"Erreur de transcription audio : {str(e)}"}), 500

    # =========================================================
    # CAS 2 : MESSAGE AVEC IMAGE → modèle vision Groq
    # =========================================================
    if has_image and image_base64:
        try:
            # Construction du message multimodal (image + texte)
            prompt_text = user_message if user_message else "Analyse et décris cette image en détail."

            vision_messages = [
                {
                    "role": "system",
                    "content": "Tu es un assistant visuel. Analyse les images avec précision et décris ce que tu vois en français."
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

            # Modèle vision de Groq
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",  # modèle vision Groq
                messages=vision_messages,
                temperature=0.7,
                max_tokens=1024,
            )

            reply = response.choices[0].message.content or "Je ne peux pas analyser cette image."

            # Ajoute à l'historique (version texte seulement)
            chat_history.append({"role": "user", "content": f"[Image envoyée] {prompt_text}"})
            chat_history.append({"role": "assistant", "content": reply})

            print(f"[IMAGE] Analyse réussie")
            return jsonify({"response": reply})

        except Exception as e:
            print(f"[IMAGE ERROR] {e}")
            # Fallback : essaie avec un autre modèle vision si disponible
            try:
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
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
                                    "text": user_message if user_message else "Décris cette image."
                                }
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                )
                reply = response.choices[0].message.content or "Je ne peux pas analyser cette image."
                return jsonify({"response": reply})
            except Exception as e2:
                print(f"[IMAGE FALLBACK ERROR] {e2}")
                return jsonify({"response": f"Erreur d'analyse d'image : {str(e2)}"}), 500

    # =========================================================
    # CAS 3 : MESSAGE TEXTE NORMAL (ou texte issu de l'audio)
    # =========================================================
    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    chat_history.append({"role": "user", "content": user_message})

    try:
        recent_history = chat_history[-10:]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Tu es un assistant utile et amical. Réponds toujours en français."},
                *recent_history
            ],
            temperature=0.7,
            max_tokens=500,
        )

        reply = response.choices[0].message.content or "Réponse vide 🤖"
        chat_history.append({"role": "assistant", "content": reply})

        # Si c'était un audio, indique la transcription dans la réponse
        if has_audio:
            reply = f"🎙️ *Transcription :* « {user_message} »\n\n{reply}"

        return jsonify({"response": reply})

    except Exception as e:
        print(f"[TEXT ERROR] {e}")
        return jsonify({"error": "Erreur serveur"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
