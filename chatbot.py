import os
import re
import base64
import tempfile
import urllib.parse
import requests
import io

from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[TTS] gTTS non installé — pip install gTTS")

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
    "logo":         "minimalist professional flat vector logo, clean sharp edges, bold colors, white background, centered composition, SVG style",
    "icon":         "simple modern app icon, flat design, bold colors, clean lines, centered, rounded corners",
    "illustration": "vibrant detailed african art illustration, rich warm colors, professional digital painting, intricate patterns",
    "photo":        "photorealistic professional DSLR photo, sharp focus, perfect lighting, 8k resolution, high detail",
    "pattern":      "beautiful seamless african kente textile pattern, vibrant traditional colors, intricate geometric weave",
    "banner":       "modern professional wide marketing banner, bold typography, vibrant gradient colors, eye-catching",
    "avatar":       "professional portrait photo, centered face, studio lighting, sharp focus, neutral background",
    "poster":       "eye-catching event poster design, bold typography, vibrant colors, dynamic composition",
    "general":      "high quality professional digital art, vibrant colors, sharp details, 4k, masterpiece",
}

TYPE_SIZES = {
    "logo": (1024,1024), "icon": (512,512), "illustration": (768,1024),
    "photo": (1024,1024), "pattern": (1024,1024), "banner": (1024,512),
    "avatar": (512,512), "poster": (768,1024), "general": (1024,1024),
}

IMAGE_TYPE_KEYWORDS = {
    "logo":         ["logo"],
    "icon":         ["icône", "icone", "icon"],
    "illustration": ["illustration", "art", "dessin", "nataal bu", "nataal yu rafet"],
    "photo":        ["photo", "photographie", "réaliste", "realistic"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu"],
    "banner":       ["bannière", "banniere", "banner", "couverture"],
    "avatar":       ["avatar", "profil", "portrait", "visage"],
    "poster":       ["affiche", "poster", "flyer"],
}

IMAGE_TRIGGERS = [
    "logo","logos","icône","icones","icon","icons","illustration","illustrations",
    "avatar","avatars","bannière","banniere","banner","banners","affiche","poster",
    "posters","flyer","flyers","motif","pattern","patterns","visuel","visuels",
    "dessin","dessins","portrait","portraits","nataal","nataalu","nataalyi","nataalye",
]

IMAGE_VERBS = [
    "génère","générer","genere","generer","crée","créer","cree","creer",
    "dessine","dessiner","fais","faire","montre","montrer","produis","produire",
    "réalise","realise","imagine","génère-moi","fais-moi","crée-moi",
    "generate","create","draw","make","render","design",
    "def","defal","bind","bindal","yëgël","yegal","def ma","bind ma","wone","woneel",
]

def _norm(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a").replace("ç","c")
        .replace("ù","u").replace("û","u").replace("î","i")
        .replace("ï","i").replace("ô","o"))


# ─────────────────────────────────────────────────────────────
# EXTRACTION DU TEXTE À ÉCRIRE — VIA LLM
# ─────────────────────────────────────────────────────────────
def extract_text_to_write_llm(user_message: str):
    """Utilise le LLM pour extraire le texte à écrire dans l'image,
    quelle que soit la formulation (français, wolof, mixte)."""
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un extracteur de texte pour génération d'images. "
                        "L'utilisateur demande de créer une image et veut parfois un texte "
                        "spécifique ÉCRIT dessus (nom du logo, slogan, titre, acronyme...).\n\n"
                        "Ta tâche : extraire UNIQUEMENT le texte à écrire dans l'image.\n\n"
                        "RÈGLES STRICTES :\n"
                        "- Si un nom/texte/slogan spécifique est mentionné → retourne-le EXACTEMENT, rien d'autre.\n"
                        "- Si aucun texte spécifique → retourne exactement le mot : NULL\n"
                        "- Ne retourne JAMAIS une phrase ou explication.\n"
                        "- Ne traduis pas, ne modifie pas.\n\n"
                        "EXEMPLES :\n"
                        "'crée un logo dont le nom est DG' → DG\n"
                        "'logo pour mon restaurant La Teranga' → La Teranga\n"
                        "'génère un logo avec écrit DAKAR STYLE' → DAKAR STYLE\n"
                        "'logo pour SenTech' → SenTech\n"
                        "'def ma yenn logo bi ci dëkk bi WAAY FC' → WAAY FC\n"
                        "'fais un logo qui dit YELEN' → YELEN\n"
                        "'logo ak texte bi DG Consulting' → DG Consulting\n"
                        "'logo avec le slogan Together We Rise' → Together We Rise\n"
                        "'fais un logo bleu' → NULL\n"
                        "'génère une illustration africaine' → NULL\n"
                        "'crée un avatar professionnel' → NULL\n"
                        "'fais un logo moderne' → NULL\n"
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=30,
        )
        result = r.choices[0].message.content.strip()
        print(f"[TEXT EXTRACT] '{user_message[:60]}' → '{result}'")
        if result.upper() == "NULL" or not result or len(result) > 60:
            return None
        return result
    except Exception as e:
        print(f"[TEXT EXTRACT ERR] {e}")
        return None


def detect_image_intent(msg: str):
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
    text_to_write = extract_text_to_write_llm(msg)
    return {
        "type":                 gen_type,
        "visual_prompt":        msg,
        "text_to_write":        text_to_write,
        "confirmation_message": "🎨 Image générée !",
    }


# ─────────────────────────────────────────────────────────────
# DÉTECTION LANGUE
# ─────────────────────────────────────────────────────────────
WOLOF_WORDS = {
    "nanga def":5,"nanga xam":5,"nanga dem":5,"mangi fi rekk":5,"maa ngi fi":5,
    "jërejëf lool":5,"baal ma ko":5,"waaw waaw":5,"dafa baax na":5,"dafa neex na":5,
    "amul solo":5,"def ma":5,"bind ma":5,"yëgël ma":5,"soo bëgg":5,"bëgg naa":5,
    "asalaa maalekum":5,"maalekum salaam":5,"dama bëgg":5,"dama dem":5,"dama nekk":5,
    "yow noo":5,"lan moy":5,"naka nga def":5,"dëgg naa":5,"fi rekk":5,"mooy":5,
    "jërejëf":4,"jërëjëf":4,"baal ma":4,"deedeet":4,"mangi fi":4,"mangi":4,
    "maa ngi":4,"nataal":4,"nataalu":4,"liggéey":4,"liggeeyu":4,
    "dafa":4,"dama":4,"xam":4,"bëgg":4,"siiw":4,"rekk":4,
    "yëgël":4,"mbokk":4,"xarit":4,"suñu":4,"sunu":4,"dëkk":4,
    "tubaab":4,"nit":4,"wolof":4,"seereer":4,"pulaar":4,"muñ":4,
    "soppi":4,"xool":4,"bëgg bëgg":4,"ñaar":4,"ñett":4,"juróom":4,"fukk":4,
    "waaw":3,"yow":3,"moom":3,"naa":3,"nga":3,"nekk":3,"wax":3,
    "gis":3,"rafet":3,"baax":3,"neex":3,"ndax":3,"waaye":3,
    "tey":3,"jaay":3,"jënd":3,"defal":3,"daldi":3,"seet":3,
    "kaay":3,"ñëw":3,"dem":3,"lekk":3,"dox":3,"bind":3,"jëf":3,
    "tëdd":3,"xaar":3,"jox":3,"naan":3,"teg":3,"fal":3,"gën":3,
    "yalla":3,"baraka":3,
    "incha allah":2,"alhamdoulilah":2,"bismillah":2,"masha allah":2,
    "ak":2,"sama":2,"seen":2,"ci":2,"bi":2,"yi":2,"bu":2,"gi":2,
    "baay":2,"yaay":2,"xale":2,"goor":2,"doom":2,
    "benn":2,"lool":2,"def":2,"am":2,"ko":2,"mu":2,"nu":2,"ñu":2,
    "di":2,"la":2,"na":2,"ni":2,"fi":2,"ba":2,"mo":2,
}

def detect_language(text: str) -> str:
    t = text.lower()
    wolof_score = sum(weight for word, weight in WOLOF_WORDS.items() if word in t)
    french_words = ["je","tu","il","elle","nous","vous","les","des","une","est",
                    "avec","bonjour","merci","comment","pourquoi","mais","donc",
                    "alors","parce","quand","pour","dans","que","qui","très","bien"]
    french_score = sum(1 for w in french_words if f" {w} " in f" {t} ")
    print(f"[LANG SCORE] wolof={wolof_score} french={french_score} | text='{t[:60]}'")
    if wolof_score >= 2:                       return "wolof"
    if wolof_score >= 1 and french_score >= 1: return "wolof"
    if french_score >= 3:                      return "french"
    return "french"

def _wolof_score(text: str) -> int:
    t = text.lower()
    return sum(weight for word, weight in WOLOF_WORDS.items() if word in t)


# ─────────────────────────────────────────────────────────────
# EXTRACTION TEXTE DOCUMENT
# ─────────────────────────────────────────────────────────────
def extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext in ("txt","md","csv","json","xml","html","py","js"):
        return file_bytes.decode("utf-8", errors="replace")[:15000]
    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "".join(p.extract_text() or "" for p in reader.pages[:20])
            return text[:15000] if text.strip() else "PDF sans texte extractible."
        except Exception as e:
            return f"Erreur PDF: {e}"
    if ext == "docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs)[:15000]
        except Exception as e:
            return f"Erreur DOCX: {e}"
    return f"❌ Format '{ext}' non supporté."

def analyze_document(doc_text, filename, question, lang):
    system = ("Tu es Yelen AI, expert en analyse de documents. Sois concis et clair. "
              + ("Réponds en français ou wolof." if lang=="wolof" else "Réponds en français."))
    q = question.strip() if question.strip() else "Fais un résumé complet."
    prompt = f'Document "{filename}" :\n\n---\n{doc_text[:12000]}\n---\n\nQuestion : {q}'
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=1000,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur analyse : {str(e)}"


# ─────────────────────────────────────────────────────────────
# TRADUCTION PROMPT IMAGE → ANGLAIS
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str, text_to_write=None) -> str:
    user_content = prompt
    if text_to_write:
        user_content += (
            f'\n\nCRITICAL: The image MUST display the exact text "{text_to_write}" '
            f'written in large, bold, clearly readable typography. '
            f'The text "{text_to_write}" must be perfectly legible and correctly spelled.'
        )
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"system","content":"You are an image generation prompt translator. Translate the user's request into a detailed English image generation prompt. Return ONLY the English prompt, no explanation."},
                {"role":"user","content":user_content},
            ],
            temperature=0.2, max_tokens=200,
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
def _looks_like_image(raw: bytes) -> bool:
    if not raw or len(raw) < 500: return False
    if raw[:8] == b'\x89PNG\r\n\x1a\n': return True
    if raw[:2] == b'\xff\xd8': return True
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP': return True
    if raw[:6] in (b'GIF87a', b'GIF89a'): return True
    return False

def generate_image(prompt: str, gen_type: str, text_to_write=None):
    english = translate_prompt_to_english(prompt, text_to_write)
    prefix  = TYPE_PROMPTS.get(gen_type, "")
    w, h    = TYPE_SIZES.get(gen_type, (1024,1024))

    if text_to_write:
        text_instruction = (
            f'the text "{text_to_write}" written in large bold clear readable letters, '
            f'typography "{text_to_write}" clearly visible and correctly spelled, '
            f'text "{text_to_write}" perfectly integrated in the design'
        )
        full = f"{prefix}, {english}, {text_instruction}, masterpiece, best quality, ultra detailed, sharp"
    else:
        full = f"{prefix}, {english}, masterpiece, best quality, ultra detailed, sharp"

    print(f"[IMAGE] type={gen_type} text='{text_to_write}' | {full[:120]}...")

    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={"Authorization":f"Bearer {TOGETHER_API_KEY}","Content-Type":"application/json"},
                json={"model":"black-forest-labs/FLUX.1-schnell-Free","prompt":full,
                      "width":min(w,1024),"height":min(h,1024),"steps":4,"n":1,"response_format":"b64_json"},
                timeout=90,
            )
            r.raise_for_status()
            b64 = r.json()["data"][0]["b64_json"]
            if b64 and _looks_like_image(base64.b64decode(b64)):
                return b64
        except Exception as e:
            print("[FLUX ERR]", e)

    try:
        enc = urllib.parse.quote(full)
        url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&enhance=true&model=flux"
        res = requests.get(url, timeout=90)
        if res.status_code == 200 and _looks_like_image(res.content):
            return base64.b64encode(res.content).decode()
        return None
    except Exception as e:
        print("[POLLINATIONS ERR]", e)
        return None

def get_mime(b64: str) -> str:
    try:
        h = base64.b64decode(b64[:20])
        if h[:4] == b'\x89PNG': return "image/png"
        if h[:2] == b'\xff\xd8': return "image/jpeg"
        if b'WEBP' in h: return "image/webp"
    except Exception:
        pass
    return "image/jpeg"


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION AUDIO
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
    suffix, mime = ".m4a", "audio/mp4"
    if len(audio_bytes) >= 4:
        if audio_bytes[:4] == b'RIFF': suffix, mime = ".wav", "audio/wav"
        elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb': suffix, mime = ".mp3", "audio/mpeg"
        elif len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp': suffix, mime = ".m4a", "audio/mp4"
        elif audio_bytes[:4] == b'OggS': suffix, mime = ".ogg", "audio/ogg"
        elif audio_bytes[:4] == b'\x1A\x45\xDF\xA3': suffix, mime = ".webm", "audio/webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    PROMPT = (
        "Nanga def? Jërejëf waay, mangi fi rekk, dama bëgg wax ak yow ci "
        "sama liggéey bi. Dafa neex na lool. Sama xarit yi ñëw ci dëkk bi, "
        "ñu ngi wax ci wolof ak français. Bul fekk, dégg naa la bu baax, waaw waaw."
    )

    def _run(language):
        kwargs = dict(model="whisper-large-v3", response_format="text", temperature=0.0, prompt=PROMPT)
        if language: kwargs["language"] = language
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(file=(f"audio{suffix}", f, mime), **kwargs)
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return (text or "").strip()

    try:
        candidates = []
        for lang in ("fr", None):
            try:
                text = _run(lang)
                if text: candidates.append(text)
            except Exception as e:
                print(f"[WHISPER lang={lang} ERR]", e)
        if not candidates: return ""
        best = max(candidates, key=_wolof_score) if len(candidates) > 1 else candidates[0]
        print(f"[WHISPER] retenu={best!r}")
        return best
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────
# CORRECTION WOLOF
# ─────────────────────────────────────────────────────────────
_WOLOF_REF = sorted({w for w,s in WOLOF_WORDS.items() if s >= 3}, key=len, reverse=True)[:120]

def correct_wolof_transcription(text: str) -> str:
    glossary = ", ".join(_WOLOF_REF)
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":(
                    "Tu es un correcteur expert de transcriptions vocales wolof/français. "
                    "Corrige UNIQUEMENT l'orthographe des mots mal transcrits. "
                    "Ne traduis JAMAIS. Ne reformule pas. Ne complète pas. "
                    "Déformations fréquentes : 'nan ga def'→'nanga def', 'jere jef'→'jërejëf', "
                    "'deug'→'dëgg', 'beug'→'bëgg', 'cham'/'kham'→'xam', 'gnew'→'ñëw', "
                    "'wow'→'waaw', 'didi'→'deedeet'. "
                    f"Référence : {glossary}. "
                    "Retourne UNIQUEMENT la transcription corrigée."
                )},
                {"role":"user","content":text},
            ],
            temperature=0, max_tokens=500,
        )
        corrected = r.choices[0].message.content.strip()
        print(f"[CORRECTION] '{text}' → '{corrected}'")
        return corrected
    except Exception as e:
        print("[CORRECTION ERR]", e)
        return text


# ─────────────────────────────────────────────────────────────
# TTS — gTTS avec phonétique wolof
# ─────────────────────────────────────────────────────────────
_WOLOF_PRONUNCIATION = {
    "jërejëf":"djérédjef","waaw":"waw","deedeet":"dédet",
    "mangi fi":"man-gui fi","maa ngi":"ma ngui",
    "dafa":"dafa","dama":"dama","bëgg":"bèg","rekk":"rèk","lool":"lol",
    "ndax":"ndak","xam":"kham","ñëw":"gnew","ñu":"gnu",
    "nekk":"nèk","dem":"dèm","lekk":"lèk","wax":"wak","gis":"guiss",
    "teg":"tèg","muñ":"mougn","soppi":"soppi","xool":"khol",
    "kaay":"kay","sama":"sama","seen":"sène","suñu":"sougnou",
    "baax":"bak","rafet":"rafète","neex":"nèkh","yëgël":"yéguèl",
    "baal ma":"bal ma","asalaa maalekum":"assalamu alaykoum",
    "maalekum salaam":"alaykoumoussalam","incha allah":"inchallah",
    "yalla":"yalla","baraka":"baraka","ë":"é","ñ":"gn",
}
_MD_RE    = re.compile(r'[*_`#>~\[\]()\|]')
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+")

def _preprocess_tts(text: str, lang: str) -> str:
    t = _EMOJI_RE.sub("", text)
    t = _MD_RE.sub("", t)
    t = re.sub(r'\s+', ' ', t).strip()
    if lang == "wolof":
        for w, p in sorted(_WOLOF_PRONUNCIATION.items(), key=lambda x: -len(x[0])):
            t = re.sub(re.escape(w), p, t, flags=re.IGNORECASE)
    return t[:1200]

def generate_speech(text: str, lang_detected: str = "french"):
    if not TTS_AVAILABLE: return None
    clean = _preprocess_tts(text, lang_detected)
    if not clean: return None
    try:
        buf = io.BytesIO()
        gTTS(text=clean, lang="fr", slow=False).write_to_fp(buf)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print("[TTS ERR]", e)
        return None


# ─────────────────────────────────────────────────────────────
# SYSTÈMES DE PROMPTS
# ─────────────────────────────────────────────────────────────
WOLOF_SYSTEM = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles couramment le wolof et le français, comme un jeune Dakarois éduqué des années 2020.

RÈGLES :
1. WOLOF PUR → réponds EN WOLOF.
2. MÉLANGÉ wolof-français → réponds dans le MÊME style.
3. FRANÇAIS PUR → réponds EN FRANÇAIS.
4. Sois NATUREL, CONCIS, CHALEUREUX.
5. N'invente JAMAIS de mots wolof inexistants.
6. Si tu ne comprends pas, demande de répéter.

GRAMMAIRE :
• Dama+verbe=Je | Dafa+verbe=Il/Elle | Mangi=Je suis en train de
• Négation : -ul | duma | dul
• Questions : Naka? | Lan? | Fan? | Ndax?

EXEMPLES :
"Nanga def?" → "Mangi fi rekk, jërejëf! Yow noo?"
"Logo bi rafet na!" → "Jërejëf lool! Danga bëgg yenn add?"
"""

FRENCH_SYSTEM = (
    "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
    "Tu réponds toujours en français. "
    "Tu peux créer des images, logos, illustrations, analyser des documents. "
    "Sois naturel, utile et positif."
)

def handle_chat(user_message: str, history: list) -> dict:
    intent = detect_image_intent(user_message)
    if intent:
        text_to_write = intent.get("text_to_write")
        img = generate_image(intent["visual_prompt"], intent["type"], text_to_write)
        if img:
            confirmation = "🎨 Image générée !"
            if text_to_write:
                confirmation = f"🎨 Image générée avec le texte « {text_to_write} » !"
            return {
                "response":      confirmation,
                "has_image":     True,
                "image_base64":  img,
                "image_type":    intent["type"],
                "visual_prompt": intent["visual_prompt"],
                "text_written":  text_to_write,
            }
        return {"response": "❌ Génération échouée. Réessaie dans quelques secondes."}

    lang   = detect_language(user_message)
    system = WOLOF_SYSTEM if lang == "wolof" else FRENCH_SYSTEM
    print(f"[LANG DETECTED] {lang} | message='{user_message[:60]}'")

    messages = [{"role":"system","content":system}]
    for msg in history[-10:]:
        if msg.get("role") in ("user","assistant") and msg.get("content"):
            messages.append({"role":msg["role"],"content":msg["content"]})
    messages.append({"role":"user","content":user_message})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages, temperature=0.7, max_tokens=600,
    )
    return {"response": r.choices[0].message.content, "_lang": lang}


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

@app.route("/tts", methods=["POST"])
def tts_route():
    data = request.get_json()
    if not data: return jsonify({"error":"invalid request"}), 400
    text = data.get("text","").strip()
    lang = data.get("lang","french")
    if not text: return jsonify({"error":"Texte vide"}), 400
    audio = generate_speech(text, lang)
    if not audio: return jsonify({"error":"Synthèse vocale indisponible"}), 500
    return jsonify({"audio_base64":audio,"has_audio_response":True})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data: return jsonify({"error":"invalid request"}), 400

    user_message = data.get("message","")
    has_image    = data.get("has_image", False)
    image_base64 = data.get("image_base64")
    has_audio    = data.get("has_audio", False)
    audio_base64 = data.get("audio_base64")
    has_document = data.get("has_document", False)
    doc_base64   = data.get("doc_base64")
    doc_filename = data.get("doc_filename","document.txt")
    history      = data.get("history", [])

    # ── 🎙 AUDIO ──
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            if len(audio_bytes) < 500:
                return jsonify({"response":"❌ Audio trop court."})

            transcribed = transcribe_audio(audio_bytes)
            transcribed = correct_wolof_transcription(transcribed)
            print("[TRANSCRIPTION FINALE]", repr(transcribed))

            if not transcribed or len(transcribed.strip()) < 2:
                return jsonify({"response":"❌ Audio non reconnu. Rapproche-toi du micro."})

            try:
                result = handle_chat(transcribed, history)
            except Exception as e:
                result = {"response": f"❌ Erreur : {str(e)}"}

            lang_detected = result.pop("_lang", detect_language(transcribed))
            # transcription envoyée séparément — la bulle user reste "🎙️ Message vocal"
            result["transcription"]    = transcribed
            result["is_voice_message"] = True

            response_text = result.get("response","")
            if response_text and not result.get("has_image"):
                audio_reply = generate_speech(response_text, lang_detected)
                if audio_reply:
                    result["audio_base64"]       = audio_reply
                    result["has_audio_response"] = True

            return jsonify(result)
        except Exception as e:
            print("[AUDIO ERR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ── 📄 DOCUMENT ──
    if has_document and doc_base64:
        try:
            doc_bytes = base64.b64decode(doc_base64)
            doc_text  = extract_text_from_document(doc_bytes, doc_filename)
            if doc_text.startswith("❌"): return jsonify({"response":doc_text})
            question = user_message.strip()
            lang     = detect_language(question) if question else "french"
            response = analyze_document(doc_text, doc_filename, question, lang)
            return jsonify({"response":response,"has_document":True,"doc_filename":doc_filename})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur document : {str(e)}"})

    # ── 🖼 IMAGE ANALYSE ──
    if has_image and image_base64:
        try:
            if "," in image_base64: image_base64 = image_base64.split(",",1)[1]
            if len(image_base64) < 100: return jsonify({"response":"❌ Image invalide."})
            mime     = get_mime(image_base64)
            question = user_message.strip() or "Décris cette image en détail en français."
            for model in ["meta-llama/llama-4-scout-17b-16e-instruct","meta-llama/llama-4-maverick-17b-128e-instruct"]:
                try:
                    r = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"user","content":[
                            {"type":"image_url","image_url":{"url":f"data:{mime};base64,{image_base64}"}},
                            {"type":"text","text":question},
                        ]}],
                        max_tokens=1024,
                    )
                    return jsonify({"response": r.choices[0].message.content})
                except Exception as e:
                    print(f"[IMG ERR] {model}: {e}")
                    continue
            return jsonify({"response":"❌ Analyse image impossible."})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # ── 💬 TEXTE ──
    if not user_message.strip():
        return jsonify({"response":"❌ Message vide."})
    try:
        result = handle_chat(user_message, history)
        result.pop("_lang", None)
        return jsonify(result)
    except Exception as e:
        print("[CHAT ERR]", e)
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Yelen AI API 🌟"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
