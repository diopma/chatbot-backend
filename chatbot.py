import os
import base64
import tempfile
import urllib.parse
import requests
import io

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from groq import Groq

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

if not GROQ_API_KEY:
    raise Exception("GROQ_API_KEY manquant")

client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────
# IMAGE STYLES & SIZES
# ─────────────────────────────────────────────────────────────
TYPE_PROMPTS = {
    "logo":         "minimalist professional flat vector logo, clean sharp edges, bold colors, white background, centered composition",
    "icon":         "simple modern app icon, flat design, bold colors, clean lines, centered, rounded corners",
    "illustration": "vibrant detailed african art illustration, rich warm colors, professional digital painting",
    "photo":        "photorealistic professional DSLR photo, sharp focus, perfect lighting, 8k resolution",
    "pattern":      "beautiful seamless african kente textile pattern, vibrant traditional colors, intricate geometric weave",
    "banner":       "modern professional wide marketing banner, bold typography, vibrant gradient colors",
    "avatar":       "professional portrait photo, centered face, studio lighting, sharp focus, neutral background",
    "poster":       "eye-catching event poster design, bold typography, vibrant colors, dynamic composition",
    "general":      "high quality professional digital art, vibrant colors, sharp details, 4k, masterpiece",
}

TYPE_SIZES = {
    "logo":         (1024, 1024),
    "icon":         (512,  512),
    "illustration": (768,  1024),
    "photo":        (1024, 1024),
    "pattern":      (1024, 1024),
    "banner":       (1024, 512),
    "avatar":       (512,  512),
    "poster":       (768,  1024),
    "general":      (1024, 1024),
}

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo"],
    "icon":         ["icône", "icone", "icon"],
    "illustration": ["illustration", "art", "dessin", "nataal"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
}

IMAGE_TRIGGERS = [
    "logo", "logos", "icône", "icones", "icon", "icons",
    "illustration", "illustrations", "avatar", "avatars",
    "bannière", "banniere", "banner", "banners",
    "affiche", "poster", "posters", "flyer", "flyers",
    "motif", "pattern", "patterns", "visuel", "visuels",
    "dessin", "dessins", "portrait", "portraits",
    # Wolof
    "nataal", "nataalu", "nataalyi", "nataalye",
    "sama nataal", "seen nataal", "yenn nataal",
]

IMAGE_VERBS = [
    "génère", "générer", "genere", "generer",
    "crée", "créer", "cree", "creer",
    "dessine", "dessiner", "fais", "faire",
    "montre", "montrer", "produis", "produire",
    "réalise", "realise", "imagine",
    "génère-moi", "fais-moi", "crée-moi",
    "generate", "create", "draw", "make", "render", "design",
    # Wolof
    "def", "defal", "bind", "bindal",
    "yëgël", "yegal", "def ma", "bind ma",
    "yokk", "yokkal", "wone", "woneel",
]

def _norm(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a").replace("ç","c")
        .replace("ù","u").replace("û","u").replace("î","i")
        .replace("ï","i").replace("ô","o"))

def detect_image_intent(msg: str) -> dict | None:
    m     = _norm(msg)
    words = m.split()
    has_noun = any(_norm(n) in words or _norm(n) in m for n in IMAGE_TRIGGERS)
    has_verb = any(_norm(v) in words or _norm(v) in m for v in IMAGE_VERBS)
    if not (has_noun or (has_verb and has_noun)):
        return None
    gen_type = "general"
    for t, keywords in IMAGE_TYPE_KEYWORDS.items():
        if any(_norm(kw) in m for kw in keywords):
            gen_type = t
            break
    return {
        "type":                 gen_type,
        "visual_prompt":        msg,
        "confirmation_message": "🎨 Image générée !",
    }

# ─────────────────────────────────────────────────────────────
# DÉTECTION LANGUE
# ─────────────────────────────────────────────────────────────
WOLOF_WORDS = {
    "nanga def": 4, "mangi fi rekk": 4, "jërejëf lool": 4,
    "maa ngi fi": 4, "dafa baax": 4, "dafa neex": 4,
    "xam naa": 4, "amul solo": 4, "def ma": 4, "bind ma": 4,
    "soo bëgg": 4, "bëgg naa": 4, "waaw waaw": 4,
    "jërejëf": 3, "baal ma": 3, "deedeet": 3,
    "mangi": 3, "maa ngi": 3, "nataal": 3,
    "liggéey": 3, "dafa": 3, "dama": 3, "xam": 3,
    "bëgg": 3, "rekk": 3, "yëgël": 3, "ndanka": 3,
    "mbokk": 3, "xarit": 3, "niit": 3, "wolof": 3,
    "waaw": 2, "yow": 2, "moom": 2, "laa": 2,
    "naa": 2, "nga": 2, "nekk": 2, "topp": 2,
    "wax": 2, "gis": 2, "rafet": 2, "baax": 2,
    "neex": 2, "ndax": 2, "waaye": 2, "tey": 2,
    "jaay": 2, "jënd": 2, "daldi": 2, "seet": 2,
    "dem": 1, "ñëw": 1, "lekk": 1, "dox": 1,
    "fëkk": 1, "bind": 1, "jëf": 1, "tëdd": 1,
    "ak": 1, "sama": 1, "seen": 1, "ci": 1,
    "bi": 1, "yi": 1, "bu": 1, "baay": 1,
    "yaay": 1, "xale": 1, "goor": 1, "benn": 1,
    "ñaar": 1, "ñett": 1, "juróom": 1, "lool": 1,
    "sunu": 1, "def": 1, "am": 1, "yalla": 2,
    "incha allah": 2, "baraka": 2,
}

def detect_language(text: str) -> str:
    t = text.lower()
    wolof_score = sum(w for m, w in WOLOF_WORDS.items() if m in t)
    french_words = [
        "je", "tu", "il", "elle", "nous", "vous", "les", "des",
        "une", "est", "avec", "bonjour", "merci", "comment",
        "pourquoi", "mais", "donc", "alors", "pour", "dans", "que",
    ]
    french_score = sum(1 for w in french_words if f" {w} " in f" {t} ")
    print(f"[LANG] wolof={wolof_score} french={french_score} | '{t[:50]}'")
    if wolof_score >= 3:                        return "wolof"
    if wolof_score >= 2 and french_score <= 3:  return "wolof"
    if wolof_score >= 1 and french_score >= 1:  return "wolof"
    if french_score >= 2:                       return "french"
    return "french"

# ─────────────────────────────────────────────────────────────
# EXTRACTION TEXTE DOCUMENT
# ─────────────────────────────────────────────────────────────
def extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    print(f"[DOC] {filename} | {len(file_bytes)} bytes | ext={ext}")

    if ext in ("txt", "md", "csv", "json", "xml", "html", "py", "js"):
        try:
            return file_bytes.decode("utf-8", errors="replace")[:15000]
        except Exception as e:
            return f"Erreur lecture : {e}"

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "".join(page.extract_text() or "" for page in reader.pages[:20])
            return text[:15000] if text.strip() else "PDF sans texte extractible."
        except ImportError:
            return "❌ pypdf non installé. Ajoute pypdf dans requirements.txt"
        except Exception as e:
            return f"Erreur PDF: {e}"

    if ext == "docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)[:15000]
        except ImportError:
            return "❌ python-docx non installé."
        except Exception as e:
            return f"Erreur DOCX: {e}"

    if ext in ("xlsx", "xls"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            text = ""
            for sheet in wb.sheetnames[:3]:
                ws = wb[sheet]
                text += f"\n--- Feuille: {sheet} ---\n"
                for row in ws.iter_rows(max_row=100, values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        text += row_text + "\n"
            return text[:15000]
        except ImportError:
            return "❌ openpyxl non installé."
        except Exception as e:
            return f"Erreur XLSX: {e}"

    return f"❌ Format '{ext}' non supporté. Formats acceptés : PDF, DOCX, TXT, CSV, XLSX."

# ─────────────────────────────────────────────────────────────
# TRADUCTION PROMPT → ANGLAIS
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Translate the user's request into a detailed English image generation prompt. Return ONLY the English prompt, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, max_tokens=120,
        )
        translated = r.choices[0].message.content.strip()
        print(f"[PROMPT EN] {translated}")
        return translated
    except Exception as e:
        print(f"[TRANSLATE ERR] {e}")
        return prompt

# ─────────────────────────────────────────────────────────────
# GÉNÉRATION IMAGE
# ─────────────────────────────────────────────────────────────
def generate_image(prompt: str, gen_type: str) -> str | None:
    english = translate_prompt_to_english(prompt)
    prefix  = TYPE_PROMPTS.get(gen_type, "")
    full    = f"{prefix}, {english}, masterpiece, best quality, ultra detailed, sharp"
    w, h    = TYPE_SIZES.get(gen_type, (1024, 1024))
    print(f"[IMAGE] {gen_type} | {full[:80]}...")

    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json={"model": "black-forest-labs/FLUX.1-schnell-Free", "prompt": full,
                      "width": min(w,1024), "height": min(h,1024), "steps": 4,
                      "n": 1, "response_format": "b64_json"},
                timeout=90,
            )
            r.raise_for_status()
            return r.json()["data"][0]["b64_json"]
        except Exception as e:
            print("[FLUX ERR]", e)

    try:
        enc = urllib.parse.quote(full)
        url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&enhance=true&model=flux"
        res = requests.get(url, timeout=90)
        return base64.b64encode(res.content).decode()
    except Exception as e:
        print("[POLLINATIONS ERR]", e)
        return None

# ─────────────────────────────────────────────────────────────
# DÉTECTION MIME
# ─────────────────────────────────────────────────────────────
def get_mime(b64: str) -> str:
    try:
        h = base64.b64decode(b64[:20])
        if h[:4] == b'\x89PNG':  return "image/png"
        if h[:2] == b'\xff\xd8': return "image/jpeg"
        if b'WEBP' in h:         return "image/webp"
    except Exception:
        pass
    return "image/jpeg"

# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION AUDIO — ✅ response_format="text" uniquement
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    # Détection format par magic bytes
    suffix, mime = ".m4a", "audio/mp4"
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b'RIFF':
            suffix, mime = ".wav", "audio/wav"
        elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            suffix, mime = ".mp3", "audio/mpeg"
        elif len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            suffix, mime = ".m4a", "audio/mp4"
        elif audio_bytes[:4] == b'OggS':
            suffix, mime = ".ogg", "audio/ogg"

    print(f"[AUDIO] format={suffix} taille={len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio{suffix}", f, mime),
                response_format="text",   # ✅ "text" seulement — Groq ne supporte PAS verbose_json
                prompt=(
                    "Transcris exactement. Message en français ou wolof sénégalais. "
                    "Wolof: nanga def, mangi fi, jërejëf, waaw, deedeet, dama, dafa, "
                    "xam, dem, ñëw, lekk, wax, nekk, bi, yi, ci, ak, sama, bëgg, "
                    "nataal, def ma, bind ma, xale, baay, yaay, xarit, mbokk, ndax, "
                    "baal ma, yow, rekk, lool, daldi, yalla, incha allah. "
                    "Tech: logo, image, avatar, créer, générer."
                ),
            )
        # response_format="text" → retourne une string directement
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        print(f"[WHISPER] '{text[:80]}'")
        return (text or "").strip()
    finally:
        os.unlink(path)

# ─────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH — réponse audio via Groq TTS
# ─────────────────────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "french") -> bytes | None:
    """
    Génère un fichier audio MP3 à partir du texte via Groq TTS.
    Retourne les bytes audio ou None si échec.
    """
    try:
        # Nettoyer le texte pour la synthèse
        clean = (text
            .replace("❌", "").replace("⏱️", "").replace("🎨", "")
            .replace("🎙️", "").replace("✅", "").replace("📎", "")
            .replace("**", "").replace("*", "").replace("#", "")
            .replace("»", "").replace("«", "")
            .strip()
        )
        if not clean:
            return None

        # Limiter à 500 caractères pour éviter timeout
        if len(clean) > 500:
            clean = clean[:497] + "..."

        # Voix selon la langue
        voice = "alloy"   # voix neutre, bonne pour fr + wolof

        response = client.audio.speech.create(
            model="playai-tts",          # modèle TTS Groq
            voice=voice,
            input=clean,
            response_format="mp3",
        )

        # Récupérer les bytes audio
        audio_bytes = response.read() if hasattr(response, "read") else bytes(response.content)
        print(f"[TTS] {len(audio_bytes)} bytes générés")
        return audio_bytes

    except Exception as e:
        print(f"[TTS ERR] {e}")
        return None

# ─────────────────────────────────────────────────────────────
# PROMPTS SYSTÈME
# ─────────────────────────────────────────────────────────────
WOLOF_SYSTEM = """Tu es Yelen AI, assistant IA sénégalais chaleureux et moderne.
Tu parles wolof et français comme un Dakarois.

RÈGLES :
1. Wolof → réponds EN WOLOF (simple, naturel, pas trop long)
2. Mélange wolof-français → réponds dans le même style
3. Français pur → réponds en français
4. N'invente JAMAIS de mots wolof inexistants
5. Sois concis et naturel
RÈGLE TRÈS IMPORTANTE :

Réponds uniquement à la demande de l'utilisateur.

N'ajoute jamais d'informations qui n'ont pas été demandées.

Ne développe pas inutilement.

Pour une salutation, réponds uniquement par une salutation.

Pour un merci, réponds uniquement par une formule de politesse.

Pour une question courte, donne une réponse courte.

N'essaie pas de poursuivre la conversation si l'utilisateur ne l'a pas demandé.
si tu ne comprends dis tu n'a pas compris ne dis pas autre chose 

WOLOF SIMPLE (utilise ces mots) :
- Salut: "Nanga def?" / "Mangi fi rekk"
- Merci: "Jërejëf" / "Jërejëf lool"
- Oui: "Waaw" / Non: "Deedeet"
- Bien: "Dafa baax" / Beau: "Dafa rafet"
- Je comprends: "Xam naa"
- D'accord: "Siiw" / "Waaw baax"
- Pas de problème: "Amul solo"

EXEMPLES :
- "Nanga def?" → "Mangi fi rekk! Yow noo?"
- "Dama bëgg logo" → "Waaw! Logo yu naan nga bëgg — restaurant, magasin walla lan?"
- "Jërejëf" → "Amul solo! Lan lañu def?"
"""

FRENCH_SYSTEM = (
    "Tu es Yelen AI, assistant IA africain intelligent et chaleureux. "
    "Réponds en français, sois concis et utile. "
    "Tu peux créer des images/logos, analyser des documents PDF/Word/Excel."
)

# ─────────────────────────────────────────────────────────────
# HANDLE CHAT
# ─────────────────────────────────────────────────────────────
def handle_chat(user_message: str, history: list) -> dict:
    # 1. Image ?
    intent = detect_image_intent(user_message)
    if intent:
        img = generate_image(intent["visual_prompt"], intent["type"])
        if img:
            return {
                "response":      intent["confirmation_message"],
                "has_image":     True,
                "image_base64":  img,
                "image_type":    intent["type"],
                "visual_prompt": intent["visual_prompt"],
            }
        return {"response": "❌ Génération échouée. Réessaie."}

    # 2. Langue
    lang   = detect_language(user_message)
    system = WOLOF_SYSTEM if lang == "wolof" else FRENCH_SYSTEM

    # 3. LLM
    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=400,
    )
    return {"response": r.choices[0].message.content, "lang": lang}

# ─────────────────────────────────────────────────────────────
# KEEP-ALIVE
# ─────────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ─────────────────────────────────────────────────────────────
# ROUTE /tts — Text to Speech séparé
# ─────────────────────────────────────────────────────────────
@app.route("/tts", methods=["POST"])
def tts_route():
    """Endpoint dédié TTS : reçoit {text, lang} → retourne audio/mpeg"""
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "text manquant"}), 400

    text = data.get("text", "")
    lang = data.get("lang", "french")

    audio_bytes = text_to_speech(text, lang)
    if not audio_bytes:
        return jsonify({"error": "TTS échoué"}), 500

    # Retourner l'audio en base64 pour le frontend mobile
    audio_b64 = base64.b64encode(audio_bytes).decode()
    return jsonify({"audio_base64": audio_b64, "format": "mp3"})

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
    has_document  = data.get("has_document", False)
    doc_base64    = data.get("doc_base64")
    doc_filename  = data.get("doc_filename", "document.txt")
    history       = data.get("history", [])
    want_tts      = data.get("tts", False)   # ← frontend demande réponse audio

    # ── 🎙 AUDIO ──
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            if len(audio_bytes) < 500:
                return jsonify({"response": "❌ Audio trop court. Parle plus longtemps."})

            transcribed = transcribe_audio(audio_bytes)
            print("[TRANSCRIPTION]", repr(transcribed))

            if not transcribed or len(transcribed.strip()) < 2:
                return jsonify({"response": "❌ Audio non reconnu. Rapproche-toi du micro."})

            result = handle_chat(transcribed, history)
            result["transcription"] = transcribed

            # TTS automatique pour les réponses vocales
            if result.get("response") and not result.get("has_image"):
                lang = result.get("lang", "french")
                audio_resp = text_to_speech(result["response"], lang)
                if audio_resp:
                    result["response_audio_b64"] = base64.b64encode(audio_resp).decode()
                    result["response_audio_format"] = "mp3"

            return jsonify(result)

        except Exception as e:
            print("[AUDIO ERR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ── 📄 DOCUMENT ──
    if has_document and doc_base64:
        try:
            doc_bytes = base64.b64decode(doc_base64)
            doc_text  = extract_text_from_document(doc_bytes, doc_filename)

            if doc_text.startswith("❌"):
                return jsonify({"response": doc_text})

            question = user_message.strip() or "Fais un résumé complet de ce document."
            lang     = detect_language(question)

            system = (
                "Tu es Yelen AI, expert en analyse de documents. "
                "Analyse le document fourni et réponds précisément. "
                f"Réponds en {'wolof et français' if lang == 'wolof' else 'français'}."
            )
            prompt = f'Document "{doc_filename}" :\n\n---\n{doc_text[:12000]}\n---\n\nQuestion : {question}'

            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return jsonify({
                "response":     r.choices[0].message.content,
                "has_document": True,
                "doc_filename": doc_filename,
            })
        except Exception as e:
            print("[DOC ERR]", e)
            return jsonify({"response": f"❌ Erreur document : {str(e)}"})

    # ── 🖼 ANALYSE IMAGE ──
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})

            mime     = get_mime(image_base64)
            question = user_message.strip() or "Décris cette image en détail en français."

            for model in [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
            ]:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_base64}"}},
                            {"type": "text", "text": question},
                        ]}],
                        max_tokens=1024,
                    )
                    return jsonify({"response": r.choices[0].message.content})
                except Exception as e:
                    print(f"[IMG ERR] {model}: {e}")
                    continue

            return jsonify({"response": "❌ Analyse image impossible."})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # ── 💬 TEXTE ──
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    try:
        result = handle_chat(user_message, history)

        # TTS si demandé et pas d'image
        if want_tts and result.get("response") and not result.get("has_image"):
            lang = result.get("lang", "french")
            audio_resp = text_to_speech(result["response"], lang)
            if audio_resp:
                result["response_audio_b64"]    = base64.b64encode(audio_resp).decode()
                result["response_audio_format"] = "mp3"

        return jsonify(result)
    except Exception as e:
        print("[CHAT ERR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
