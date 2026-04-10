import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Charger variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# 🔐 Rate limit (anti spam)
limiter = Limiter(get_remote_address, app=app)

# 🔐 Clé Groq
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise Exception("Définir GROQ_API_KEY")

client = Groq(api_key=groq_api_key)

chat_history = []

# 🔐 Token dynamique (change chaque minute)
def generate_token():
    return str(int(time.time() / 60))

@app.route("/")
def home():
    return "Serveur sécurisé 🚀"

@app.route("/chat", methods=["POST"])
@limiter.limit("10 per minute")  # 🔥 anti spam
def chat():
    global chat_history

    # 🔐 Vérifier token
    client_token = request.headers.get("x-token")
    if client_token != generate_token():
        return jsonify({"error": "Token invalide 🔒"}), 403

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
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
