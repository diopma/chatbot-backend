import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

app = Flask(__name__)
CORS(app)

# 🔐 Clé Groq
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise Exception("Définir GROQ_API_KEY dans vos variables d'environnement.")

client = Groq(api_key=groq_api_key)

# 🔐 Clé secrète pour sécuriser ton API
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise Exception("Définir API_KEY dans vos variables d'environnement.")

# Historique temporaire
chat_history = []

# Route test
@app.route("/", methods=["GET"])
def home():
    return "Serveur Groq STABLE 🚀"

# Route chat sécurisée
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # 🔐 Vérification de la clé API
    client_key = request.headers.get("x-api-key")
    if client_key != API_KEY:
        return jsonify({"error": "Accès refusé 🔒"}), 403

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message invalide"}), 400

    user_message = data["message"]
    chat_history.append({"role": "user", "content": user_message})

    try:
        # Limite historique
        recent_history = chat_history[-10:]

        # Appel Groq
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

# Lancement serveur
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
