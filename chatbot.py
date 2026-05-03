import os
import base64
import tempfile
import urllib.parse
import json
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY manquant")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────
# IMAGE STYLES & SIZES
# ─────────────────────────────────────────────────────────────
TYPE_PROMPTS = {
    "logo":         "minimalist flat vector logo, clean design, centered",
    "icon":         "simple app icon, flat design",
    "illustration": "african art illustration, colorful, detailed",
    "photo":        "photorealistic DSLR quality",
    "pattern":      "african textile seamless pattern",
    "banner":       "modern wide banner design",
    "avatar":       "profile portrait centered face",
    "poster":       "event poster, bold typography",
    "general":      "high quality digital art",
}

TYPE_SIZES = {
    "logo":         (1024, 1024),
    "icon":         (512,  512),
    "illustration": (768,  1024),
    "photo":        (768,  768),
    "pattern":      (1024, 1024),
    "banner":       (1200, 630),
    "avatar":       (512,  512),
    "poster":       (768,  1024),
    "general":      (768,  768),
}

# ─────────────────────────────────────────────────────────────
# DÉTECTION IMAGE
# Logique : verbe d'action OU mot visuel direct dans le message
# ─────────────────────────────────────────────────────────────

# Verbes qui indiquent une intention de création
ACTION_VERBS = [
    "génère", "générer", "genere", "generer",
    "crée", "créer", "cree", "creer",
    "dessine", "dessiner",
    "fais", "faire", "fais-moi", "faisмне",
    "montre", "montrer",
    "produis", "produire",
    "réalise", "realise",
    "generate", "create", "draw", "make", "render", "produce", "design", "imagine",
]

# Mots visuels qui seuls suffisent à déclencher la génération
VISUAL_NOUNS = [
    "logo", "logos",
    "icône", "icones", "icon", "icons",
    "illustration", "illustrations",
    "avatar", "avatars",
    "bannière", "banniere", "banner", "banners",
    "affiche", "poster", "posters", "flyer", "flyers",
    "motif", "pattern", "patterns",
    "image", "images", "photo", "photos",
    "dessin", "dessins",
    "visuel", "visuels",
    "portrait", "portraits",
]

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo"],
    "icon":         ["icône", "icone", "icon"],
    "illustration": ["illustration", "art", "dessin"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
}

def quick_detect_image(message: str) -> bool:
    """
    Détecte si le message demande une image.
    Deux cas :
    1. Verbe d'action présent dans le message
    2. Nom visuel seul (ex: "logo pour mon resto")
    """
    msg = message.lower()
    # Nettoie les accents pour comparaison
    msg_clean = (msg
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a")
        .replace("ù","u").replace("û","u")
        .replace("î","i").replace("ï","i")
        .replace("ô","o").replace("ç","c")
    )

    words = msg_clean.split()

    # Cas 1 : verbe d'action présent
    for verb in ACTION_VERBS:
        verb_clean = (verb
            .replace("é","e").replace("è","e").replace("ê","e")
            .replace("à","a").replace("ç","c")
        )
        if verb_clean in words or verb_clean in msg_clean:
            print(f"[DETECT] Verbe trouvé: {verb}")
            return True

    # Cas 2 : nom visuel seul dans le message (ex: "logo pour mon entreprise")
    for noun in VISUAL_NOUNS:
        if noun in words:
            print(f"[DETECT] Nom visuel trouvé: {noun}")
            return True

    return False

def quick_detect_type(message: str) -> str:
    msg = message.lower()
    for img_type, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(kw in msg for kw in keywords):
            return img_type
    return "general"

# ─────────────────────────────────────────────────────────────
# DÉTECTION IA
# ─────────────────────────────────────────────────────────────
def detect_image_intent(user_message: str) -> dict | None:
    # Étape 1 : filtre rapide par mots-clés
    if not quick_detect_image(user_message):
        return None

    # Étape 2 : confirmation par LLM léger
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un détecteur d'intention. Réponds UNIQUEMENT en JSON valide, sans markdown."
                },
                {
                    "role": "user",
                    "content": (
                        f"Ce message demande-t-il de créer/générer une image, logo, illustration ou visuel ?\n"
                        f"Message: \"{user_message}\"\n\n"
                        f"Si OUI: {{\"is_image_request\":true,\"type\":\"logo|icon|illustration|photo|pattern|banner|avatar|poster|general\","
                        f"\"visual_prompt\":\"description en anglais détaillée\",\"confirmation_message\":\"message de confirmation en français\"}}\n"
                        f"Si NON: {{\"is_image_request\":false}}"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=200,
        )

        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()

        # Extrait le JSON même si du texte parasite est présent
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        data = json.loads(raw)
        if data.get("is_image_request") is True:
            print(f"[IMAGE INTENT] type={data.get('type')} prompt={data.get('visual_prompt','')[:50]}")
            return data
        return None

    except Exception as e:
        print("[INTENT ERROR]", e)
        # Fallback direct si LLM échoue
        return {
            "is_image_request":     True,
            "type":                 quick_detect_type(user_message),
            "visual_prompt":        user_message,
            "confirmation_message": "🎨 Génération en cours...",
        }

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    prefix      = TYPE_PROMPTS.get(gen_type, "")
    full_prompt = f"{prefix}, high quality, {prompt}"
    width, height = TYPE_SIZES.get(gen_type, (768, 768))

    # ── Together AI (FLUX) ──
    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":           "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt":          full_prompt,
                    "width":           min(width, 1024),
                    "height":          min(height, 1024),
                    "steps":           4,
                    "n":               1,
                    "response_format": "b64_json",
                },
                timeout=90,
            )
            r.raise_for_status()
            print("[IMAGE GEN] FLUX OK")
            return r.json()["data"][0]["b64_json"]
        except Exception as e:
            print("[FLUX ERROR]", e)

    # ── Fallback Pollinations ──
    try:
        encoded = urllib.parse.quote(full_prompt)
        url     = f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true"
        img     = requests.get(url, timeout=90)
        print("[IMAGE GEN] Pollinations OK")
        return base64.b64encode(img.content).decode()
    except Exception as e:
        print("[POLLINATIONS ERROR]", e)
        return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION MIME
# ─────────────────────────────────────────────────────────────
def get_image_media_type(image_base64: str) -> str:
    try:
        header = base64.b64decode(image_base64[:20])
        if header[:4] == b'\x89PNG':    return "image/png"
        elif header[:2] == b'\xff\xd8': return "image/jpeg"
        elif b'WEBP'   in header:       return "image/webp"
        elif header[:4] == b'GIF8':     return "image/gif"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# ROUTE /chat
# ─────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    user_message  = data.get("message", "")
    has_image     = data.get("has_image", False)
    image_base64  = data.get("image_base64")
    has_audio     = data.get("has_audio", False)
    audio_base64  = data.get("audio_base64")
    history       = data.get("history", [])  # historique envoyé par le frontend

    # ─────────────────────────────
    # 🎙 AUDIO → TRANSCRIPTION
    # ─────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",  # rapide
                    file=("audio.wav", f, "audio/wav"),
                    response_format="text",           # pas de language forcé → auto-détection
                )
            os.unlink(path)

            user_message = transcription if isinstance(transcription, str) else transcription.text
            print("[TRANSCRIPTION]", repr(user_message))

            if not user_message or not user_message.strip():
                return jsonify({"response": "❌ Audio non reconnu. Parle plus clairement."})

            # Continue vers la détection image ou chat

        except Exception as e:
            print("[AUDIO ERROR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ─────────────────────────────
    # 🖼 ANALYSE IMAGE
    # ─────────────────────────────
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            media_type = get_image_media_type(image_base64)
            question   = user_message.strip() if user_message.strip() else "Décris cette image en détail en français."

            print(f"[IMAGE ANALYSIS] media_type={media_type}")

            r = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                        {"type": "text",      "text": question},
                    ],
                }],
                max_tokens=1024,
            )
            return jsonify({"response": r.choices[0].message.content})

        except Exception as e:
            print("[IMAGE ANALYSIS ERROR]", e)
            # Essai modèle alternatif
            try:
                r = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                            {"type": "text",      "text": question},
                        ],
                    }],
                    max_tokens=1024,
                )
                return jsonify({"response": r.choices[0].message.content})
            except Exception as e2:
                print("[IMAGE FALLBACK ERROR]", e2)
                return jsonify({"response": f"❌ Analyse image impossible : {str(e)}"})

    # ─────────────────────────────
    # 🎨 GÉNÉRATION IMAGE
    # ─────────────────────────────
    if user_message.strip():
        intent = detect_image_intent(user_message)
        if intent:
            img = generate_image(intent["visual_prompt"], intent["type"])
            if img:
                return jsonify({
                    "response":      intent["confirmation_message"],
                    "has_image":     True,
                    "image_base64":  img,
                    "image_type":    intent["type"],
                    "visual_prompt": intent["visual_prompt"],
                })
            return jsonify({"response": "❌ Génération échouée. Réessaie dans quelques secondes."})

    # ─────────────────────────────
    # 💬 CHAT NORMAL
    # ─────────────────────────────
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    messages = [
        {
            "role":    "system",
            "content": (
                "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
                "Tu réponds en français sauf si l'utilisateur écrit dans une autre langue. "
                "Tu ne prétends JAMAIS ne pas pouvoir créer d'images ou de logos — "
                "si l'utilisateur demande une image, tu lui dis de formuler sa demande "
                "avec des mots comme 'crée un logo', 'génère une image', etc."
            ),
        }
    ]

    # Historique (max 12 messages = 6 échanges)
    for msg in history[-12:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        reply = r.choices[0].message.content
        return jsonify({"response": reply})

    except Exception as e:
        print("[CHAT ERROR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
