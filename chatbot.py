import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)

# 🔐 Clé API sécurisée (Render Environment Variable)
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise Exception("Définir API_KEY dans Render !")

# 🔑 Clé Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise Exception("Définir GROQ_API_KEY !")

client = Groq(api_key=GROQ_API_KEY)

chat_history = []

@app.route("/")
def home():
    return "Serveur sécurisé 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # 🔐 Vérification API KEY
    client_key = request.headers.get("x-api-key")

    if not client_key:
        return jsonify({"error": "Clé API manquante"}), 401

    if client_key != API_KEY:
        return jsonify({"error": "Clé API invalide"}), 403

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message invalide"}), 400

    user_message = data["message"]
    chat_history.append({"role": "user", "content": user_message})

    try:
        recent_history = chat_history[-10:]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Tu es un assistant utile et amical."},
                *recent_history
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = response.choices[0].message.content or "Réponse vide 🤖"

        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})

    except Exception as e:
        print("ERREUR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
