import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env (optionnel)
load_dotenv()

app = Flask(__name__)
CORS(app)

# Lire la clé Groq depuis l'environnement
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise Exception("Définir GROQ_API_KEY dans vos variables d'environnement.")

client = Groq(api_key=api_key)

# Historique du chat (mémoire temporaire, pour un seul utilisateur)
chat_history = []

# Route racine pour tester que le serveur est en ligne
@app.route("/", methods=["GET"])
def home():
    return "Serveur Groq STABLE 🚀"

# Route pour le chat
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message invalide"}), 400

    user_message = data["message"]
    chat_history.append({"role": "user", "content": user_message})

    try:
        # Limite l'historique aux 10 derniers messages
        recent_history = chat_history[-10:]

        # Appel à Groq
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": "Tu es un assistant utile et amical."}, *recent_history],
            temperature=0.7,
            max_tokens=500
        )

        # Récupérer la réponse du bot
        reply = response.choices[0].message.content or "Réponse vide 🤖"
        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})

    except Exception as e:
        print("ERREUR:", str(e))
        return jsonify({"error": str(e)}), 500

# Démarrage du serveur compatible Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
