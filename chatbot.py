import os
import base64
import tempfile
import urllib.parse
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import requests

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GROQ_API_KEY:
    raise Exception("Définir GROQ_API_KEY !")

client = Groq(api_key=GROQ_API_KEY)
chat_history = []

# ── Prompt système spécialisé par type ──────────────────────────────────────
TYPE_PROMPTS = {
    "logo": (
        "Professional minimalist logo design, flat vector style, clean geometric shapes, "
        "transparent background, brand identity, SVG-like quality, bold typography optional, "
        "centered composition, high contrast"
    ),
    "icon": (
        "Simple flat app icon, bold centered shape, solid background color, "
        "minimal details, icon design system style, clean edges"
    ),
    "illustration": (
        "Vibrant digital illustration, African art style, colorful, detailed, "
        "poster quality, warm tones, artistic"
    ),
    "photo": (
        "Photorealistic, high resolution, professional photography, "
        "natural lighting, sharp focus, DSLR quality"
    ),
    "pattern": (
        "Seamless repeating pattern, African textile design, kente wax bogolan style, "
        "geometric ornamental, symmetrical, colorful"
    ),
    "banner": (
        "Social media banner, wide format 16:9, professional graphic design, "
        "modern layout, bold visuals, clean composition"
    ),
    "avatar": (
        "Profile picture portrait, centered face or character, circular crop friendly, "
        "clean background, high detail, professional"
    ),
    "poster": (
        "Event poster design, bold typography space, artistic composition, "
        "high contrast colors, print quality"
    ),
    "general": "",
}

TYPE_SIZES = {
    "logo":        (1024, 1024),
    "icon":        (512,  512),
    "illustration":(768,  1024),
    "photo":       (768,  768),
    "pattern":     (1024, 1024),
    "banner":      (1200, 630),
    "avatar":      (512,  512),
    "poster":      (768,  1024),
    "general":     (768,  768),
}


# ──────────────────────────────────────────────────────────────────────────────
#  DÉTECTION D'INTENTION : est-ce une demande de génération visuelle ?
# ──────────────────────────────────────────────────────────────────────────────
def detect_image_intent(user_message: str) -> dict | None:
    """
    Demande à LLaMA d'analyser le message et de retourner JSON si c'est une
    demande de génération visuelle. Retourne None sinon.
    """
    detection_prompt = f"""Analyse ce message et détermine s'il s'agit d'une demande de génération d'image, logo, icône, illustration, photo, motif, bannière, avatar ou poster.

Message : "{user_message}"

Si c'est une demande de création visuelle, réponds UNIQUEMENT avec ce JSON (sans markdown, sans explication) :
{{
  "is_image_request": true,
  "type": "logo|icon|illustration|photo|pattern|banner|avatar|poster|general",
  "visual_prompt": "description précise en anglais de ce qu'il faut générer",
  "confirmation_message": "message court en français confirmant ce que tu vas créer"
}}

Si ce n'est PAS une demande visuelle, réponds uniquement :
{{"is_image_request": false}}

Exemples de demandes visuelles :
- "crée moi un logo pour mon restaurant"
- "génère une photo d'un baobab"
- "je veux une icône d'application"
- "fais moi une illustration d'une femme en wax"
- "dessine un motif africain"
- "je voudrais un avatar"
- "can you make a banner for my business"
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": detection_prompt}],
            max_tokens=300,
            temperature=0.1,  # très déterministe pour la détection
        )
        raw = resp.choices[0].message.content.strip()
        # Nettoie les backticks si présents
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        if result.get("is_image_request"):
            return result
        return None
    except Exception as e:
        print(f"[INTENT] ❌ Détection échouée: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  GÉNÉRATION D'IMAGE
# ──────────────────────────────────────────────────────────────────────────────
def generate_image_from_prompt(visual_prompt: str, gen_type: str) -> str | None:
    """Génère une image et retourne le base64. None si échec."""

    type_prefix = TYPE_PROMPTS.get(gen_type, "")
    full_prompt = f"{type_prefix}, {visual_prompt}".strip(", ")
    width, height = TYPE_SIZES.get(gen_type, (768, 768))

    print(f"[GEN] Type={gen_type} | {full_prompt[:80]}…")

    # ── Together AI (FLUX) ──
    if TOGETHER_API_KEY:
        try:
            resp = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json",
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
            resp.raise_for_status()
            b64 = resp.json()["data"][0]["b64_json"]
            print("[GEN] ✅ Together AI (FLUX)")
            return b64
        except Exception as e:
            print(f"[GEN] ❌ Together AI: {e}")

    # ── Pollinations AI (fallback gratuit) ──
    try:
        seed    = abs(hash(full_prompt)) % 99999
        encoded = urllib.parse.quote(full_prompt)
        model_param = "&model=flux" if gen_type in ("logo", "icon", "pattern") else ""
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width={width}&height={height}&seed={seed}&nologo=true&enhance=true{model_param}"
        )
        img_resp = requests.get(url, timeout=90)
        img_resp.raise_for_status()
        b64 = base64.b64encode(img_resp.content).decode("utf-8")
        print("[GEN] ✅ Pollinations AI")
        return b64
    except Exception as e:
        print(f"[GEN] ❌ Pollinations: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  ROUTE PRINCIPALE : /chat
# ──────────────────────────────────────────────────────────────────────────────
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

    # ── CAS 1 : AUDIO → transcription ────────────────────────────────────────
    if has_audio and audio_base64:
        tmp_path = None
        transcribed_text = None
        try:
            audio_bytes = base64.b64decode(audio_base64)
            formats = (
                [(".wav","audio/wav"),(".mp3","audio/mpeg"),(".m4a","audio/m4a")]
                if audio_ext == "wav"
                else [(".m4a","audio/m4a"),(".wav","audio/wav"),(".mp3","audio/mpeg")]
            )
            for ext, mime in formats:
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        tmp.write(audio_bytes); tmp_path = tmp.name
                    with open(tmp_path, "rb") as f:
                        t = client.audio.transcriptions.create(
                            model="whisper-large-v3", file=(f"audio{ext}", f, mime),
                            language="fr", response_format="verbose_json", temperature=0.0,
                        )
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path); tmp_path = None
                    if hasattr(t, "text") and t.text.strip():
                        transcribed_text = t.text.strip(); break
                except Exception as fe:
                    print(f"[AUDIO] ❌ {ext}: {fe}")
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path); tmp_path = None
            if not transcribed_text:
                return jsonify({"response": "❌ Transcription impossible. Parlez plus fort."})
            user_message = transcribed_text
        except Exception as e:
            return jsonify({"response": f"Erreur audio : {e}"}), 500

    # ── CAS 2 : IMAGE (analyse visuelle) ─────────────────────────────────────
    if has_image and image_base64:
        try:
            prompt_text = user_message or "Décris cette image."
            msgs = [
                {"role": "system", "content": "Tu es un assistant visuel. Réponds en français."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_type};base64,{image_base64}"}},
                    {"type": "text", "text": prompt_text},
                ]}
            ]
            reply = None
            for model in ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.2-11b-vision-preview"]:
                try:
                    r = client.chat.completions.create(model=model, messages=msgs, temperature=0.7, max_tokens=1024)
                    reply = r.choices[0].message.content; break
                except: continue
            chat_history.append({"role":"user","content":f"[Image] {prompt_text}"})
            chat_history.append({"role":"assistant","content": reply or "Analyse impossible."})
            return jsonify({"response": reply or "Impossible d'analyser cette image."})
        except Exception as e:
            return jsonify({"response": f"Erreur image: {e}"}), 500

    # ── CAS 3 : TEXTE ─────────────────────────────────────────────────────────
    if not user_message:
        return jsonify({"error": "Message vide"}), 400

    # 🔍 Détection d'intention : génération visuelle ?
    intent = detect_image_intent(user_message)

    if intent:
        gen_type   = intent.get("type", "general")
        vis_prompt = intent.get("visual_prompt", user_message)
        confirm    = intent.get("confirmation_message", f"Je génère votre {gen_type}…")

        print(f"[INTENT] ✅ Demande visuelle détectée: type={gen_type}")

        b64 = generate_image_from_prompt(vis_prompt, gen_type)

        if b64:
            return jsonify({
                "response":        confirm,
                "has_image":       True,
                "image_base64":    b64,
                "image_type":      gen_type,
                "visual_prompt":   vis_prompt,
            })
        else:
            return jsonify({"response": "❌ Je n'ai pas pu générer l'image. Réessayez."})

    # Conversation normale
    chat_history.append({"role": "user", "content": user_message})
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es KoraChat, un assistant IA africain intelligent et amical. "
                        "Tu peux avoir des conversations, analyser des images et générer des visuels "
                        "(logos, icônes, illustrations, photos, motifs…) sur demande. "
                        "Réponds toujours en français, de manière claire et chaleureuse."
                    )
                },
                *chat_history[-10:]
            ],
            temperature=0.7,
            max_tokens=800,
        )
        reply = r.choices[0].message.content or "Réponse vide 🤖"
        chat_history.append({"role": "assistant", "content": reply})
        if has_audio:
            reply = f"🎙️ *« {user_message} »*\n\n{reply}"
        return jsonify({"response": reply})
    except Exception as e:
        print(f"[TEXT] ❌ {e}")
        return jsonify({"error": "Erreur serveur"}), 500


# ──────────────────────────────────────────────────────────────────────────────
#  ROUTE /generate-image (gardée pour compatibilité)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/generate-image", methods=["POST"])
def generate_image_route():
    data = request.get_json()
    prompt   = data.get("prompt", "").strip()
    gen_type = data.get("type", "general")
    if not prompt:
        return jsonify({"error": "Prompt manquant"}), 400
    b64 = generate_image_from_prompt(prompt, gen_type)
    if b64:
        return jsonify({"image_base64": b64, "source": "ok"})
    return jsonify({"error": "Génération échouée"}), 500


@app.route("/")
def home():
    return "KoraChat API 🚀"


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
