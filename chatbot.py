import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from jose import jwt

app = Flask(__name__)
CORS(app)

# 🔐 ENV VARIABLES
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_JWT_SECRET:
    raise Exception("Missing SUPABASE_JWT_SECRET")

if not GROQ_API_KEY:
    raise Exception("Missing GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

chat_history = []

# 🔐 VERIFY JWT
def verify_token(token):
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"]
        )
        return payload
    except Exception as e:
        print("JWT ERROR:", e)
        return None

@app.route("/", methods=["GET"])
def home():
    return "API RUNNING 🚀"

# 🔥 CHAT ROUTE
@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # 🔐 AUTH HEADER
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return jsonify({"error": "Token manquant"}), 401

    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Format token invalide"}), 401

    token = auth_header.split(" ")[1]

    user = verify_token(token)

    if not user:
        return jsonify({"error": "Token invalide"}), 403

    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Message invalide"}), 400

    user_message = data["message"]
    chat_history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Tu es un assistant utile et amical."},
                *chat_history[-10:]
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = response.choices[0].message.content or "Réponse vide 🤖"

        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})

    except Exception as e:
        print("ERREUR GROQ:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
