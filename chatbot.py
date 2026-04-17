import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from jose import jwt

app = Flask(__name__)
CORS(app)

# 🔑 Supabase config
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# 🔑 Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

chat_history = []

# 🔐 Vérifier JWT
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

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history

    # 🔐 récupérer token
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        return jsonify({"error": "Token manquant"}), 401

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
                {"role": "system", "content": "Tu es un assistant utile."},
                *chat_history[-10:]
            ]
        )

        reply = response.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})

    except Exception as e:
        print("ERREUR:", str(e))
        return jsonify({"error": str(e)}), 500
