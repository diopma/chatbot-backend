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
    "illustration": ["illustration", "art", "dessin", "nataal bu", "nataal yu rafet"],
    "photo":        ["photo", "photographie", "réaliste", "realistic", "nataal bu dëkk"],
    "pattern":      ["motif", "pattern", "kente", "wax", "textile", "tissu", "mbañ"],
    "banner":       ["bannière", "banniere", "banner", "couverture", "nataal bu bon"],
    "avatar":       ["avatar", "profil", "portrait", "visage", "seen bët", "sama bët"],
    "poster":       ["affiche", "poster", "flyer", "nataal bu liggéey"],
}

IMAGE_TRIGGERS = [
    "logo", "logos", "icône", "icones", "icon", "icons",
    "illustration", "illustrations", "avatar", "avatars",
    "bannière", "banniere", "banner", "banners",
    "affiche", "poster", "posters", "flyer", "flyers",
    "motif", "pattern", "patterns", "visuel", "visuels",
    "dessin", "dessins", "portrait", "portraits",
    "nataal", "nataalu", "nataalyi", "nataalye",
    "sama nataal", "seen nataal", "yenn nataal",
    "nataal bu rafet", "nataal bu baax",
    "liggéey bu nataal", "liggeeyu nataal",
]

IMAGE_VERBS = [
    "génère", "générer", "genere", "generer",
    "crée", "créer", "cree", "creer",
    "dessine", "dessiner", "fais", "faire",
    "montre", "montrer", "produis", "produire",
    "réalise", "realise", "imagine",
    "génère-moi", "fais-moi", "crée-moi",
    "generate", "create", "draw", "make", "render", "design",
    "def", "defal", "deflu",
    "bind", "bindal", "bindaale",
    "yëgël", "yegal", "yëgëlal",
    "def ma", "bind ma", "yëgël ma",
    "yokk", "yokkal", "am", "amal",
    "teg", "tegal", "daldi def", "daldi bind",
    "seet", "seetal", "wone", "woneel",
]

# ─────────────────────────────────────────────────────────────
# EXTRACTION DU TEXTE À ÉCRIRE DANS L'IMAGE
# ─────────────────────────────────────────────────────────────
# Patterns pour détecter "dont le nom est X", "avec le texte X", "écris X", etc.
_TEXT_IN_IMAGE_PATTERNS = [
    r"(?:dont\s+le\s+nom\s+est|du\s+nom\s+de|intitulé|nommé|qui\s+s'appelle|avec\s+le\s+nom)\s+[«\"']?([A-Za-z0-9À-ÿ &._\-]+)[»\"']?",
    r"(?:avec\s+le\s+texte|qui\s+écrit|avec\s+l[ae]\s+(?:mention|inscription|texte)|portant\s+le\s+texte|écris(?:\s+dessus)?|avec\s+écrit)\s+[«\"']?([A-Za-z0-9À-ÿ &._\-]+)[»\"']?",
    r"(?:le\s+nom|le\s+titre|le\s+mot|le\s+slogan|la\s+mention)\s+[«\"']([A-Za-z0-9À-ÿ &._\-]+)[»\"']",
    r"[«\"']([A-Za-z0-9À-ÿ &._\-]{2,30})[»\"']\s+(?:écrit|inscrit|dessus|dedans|au\s+(?:centre|milieu|dessus|dessous))",
    r"(?:nom|texte|titre|slogan)\s*:\s*[«\"']?([A-Za-z0-9À-ÿ &._\-]+)[»\"']?",
    # wolof
    r"(?:xam\s+ne|bi\s+def\s+ko|mu\s+bind)\s+[«\"']?([A-Za-z0-9À-ÿ &._\-]+)[»\"']?",
    r"(?:ligge\s+ak|ci\s+dëkk\s+bi)\s+[«\"']?([A-Za-z0-9À-ÿ &._\-]+)[»\"']?",
]

def extract_text_to_write(msg: str) -> str | None:
    """Extrait le texte que l'utilisateur veut voir ÉCRIT dans l'image."""
    for pattern in _TEXT_IN_IMAGE_PATTERNS:
        m = re.search(pattern, msg, re.IGNORECASE)
        if m:
            found = m.group(1).strip()
            if found:
                return found
    return None


def _norm(text: str) -> str:
    return (text.lower()
        .replace("é","e").replace("è","e").replace("ê","e")
        .replace("à","a").replace("â","a").replace("ç","c")
        .replace("ù","u").replace("û","u").replace("î","i")
        .replace("ï","i").replace("ô","o"))


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

    # Extrait le texte à écrire dans l'image (nom du logo, slogan, etc.)
    text_to_write = extract_text_to_write(msg)

    return {
        "type":                 gen_type,
        "visual_prompt":        msg,
        "text_to_write":        text_to_write,   # None si rien à écrire
        "confirmation_message": "🎨 Image générée !",
    }


# ─────────────────────────────────────────────────────────────
# DÉTECTION LANGUE
# ─────────────────────────────────────────────────────────────
WOLOF_WORDS = {
    "nanga def": 5, "nanga xam": 5, "nanga dem": 5,
    "mangi fi rekk": 5, "maa ngi fi": 5, "mangi dem": 5,
    "jërejëf lool": 5, "baal ma ko": 5, "waaw waaw": 5,
    "dafa baax na": 5, "dafa neex na": 5, "dafa rafet na": 5,
    "dafa mel ni": 5, "xam naa": 5, "faamaak": 5,
    "amul solo": 5, "def ma": 5, "bind ma": 5,
    "yëgël ma": 5, "soo bëgg": 5, "bëgg naa": 5,
    "lu baax": 5, "lu neex": 5, "maa ngi dem": 5,
    "asalaa maalekum": 5, "maalekum salaam": 5,
    "dama bëgg": 5, "dama dem": 5, "dama nekk": 5,
    "yow noo": 5, "lan moy": 5, "fan nga dem": 5,
    "naka nga def": 5, "naka waay": 5,
    "bul fekk": 5, "bul ko wax": 5,
    "dëgg naa": 5, "dégg naa": 5,
    "fii rekk": 5, "fi rekk": 5,
    "mooy": 5, "mooye": 5,
    "loolu moy": 5, "lool la": 5,
    "waxoon na": 5, "waxoon naa": 5,
    "liggéeyal ma": 5, "liggéeyal": 5, "sëriñ": 5,
    "jërejëf": 4, "jërëjëf": 4, "baal ma": 4,
    "deedeet": 4, "mangi fi": 4, "mangi": 4,
    "maa ngi": 4, "nataal": 4, "nataalu": 4,
    "liggéey": 4, "liggeeyu": 4,
    "dafa": 4, "dama": 4, "xam": 4,
    "bëgg": 4, "siiw": 4, "rekk": 4,
    "yëgël": 4, "woneel": 4, "wone": 4,
    "ndanka": 4, "ndanka ndanka": 4,
    "mbokk": 4, "xarit": 4,
    "suñu": 4, "sunu": 4,
    "dëkk": 4, "xeex": 4,
    "tubaab": 4, "nit": 4, "niit": 4,
    "wolof": 4, "seereer": 4, "pulaar": 4,
    "muñ": 4, "muñël": 4,
    "tëgg": 4, "tëgël": 4, "fëkk": 4,
    "soppi": 4, "soppil": 4,
    "xool": 4, "xoolël": 4,
    "bëgg bëgg": 4,
    "ñaar": 4, "ñett": 4, "ñent": 4,
    "juróom": 4, "fukk": 4, "téeméer": 4, "junni": 4,
    "waaw": 3, "yow": 3, "moom": 3,
    "laa": 3, "naa": 3, "nga": 3,
    "nekk": 3, "topp": 3, "wax": 3,
    "gis": 3, "gisul": 3,
    "rafet": 3, "baax": 3, "neex": 3,
    "ndax": 3, "waaye": 3, "mbaa": 3,
    "tey": 3, "bëccëk": 3, "guddi": 3,
    "jaay": 3, "jënd": 3, "defal": 3,
    "daldi": 3, "seet": 3,
    "leen": 3, "fanaan": 3,
    "kaay": 3, "kaaye": 3,
    "ñëw": 3, "dem": 3,
    "lekk": 3, "dox": 3,
    "bind": 3, "jëf": 3,
    "tëdd": 3, "xaar": 3,
    "jox": 3, "naan": 3,
    "teg": 3, "fal": 3, "gën": 3,
    "sol": 3, "door": 3,
    "yalla": 3, "baraka": 3,
    "incha allah": 2, "alhamdoulilah": 2, "bismillah": 2, "masha allah": 2,
    "ak": 2, "sama": 2, "seen": 2,
    "ci": 2, "bi": 2, "yi": 2,
    "bu": 2, "gi": 2, "ki": 2,
    "baay": 2, "yaay": 2, "xale": 2,
    "goor": 2, "doom": 2,
    "benn": 2, "lool": 2, "def": 2,
    "am": 2, "ko": 2,
    "mu": 2, "nu": 2, "ñu": 2,
    "di": 2, "la": 2, "na": 2, "ni": 2, "fi": 2,
    "ba": 2, "mo": 2,
}

def detect_language(text: str) -> str:
    t = text.lower()
    wolof_score = sum(weight for word, weight in WOLOF_WORDS.items() if word in t)
    french_words = [
        "je", "tu", "il", "elle", "nous", "vous",
        "les", "des", "une", "est", "avec", "bonjour",
        "merci", "comment", "pourquoi", "mais", "donc",
        "alors", "parce", "quand", "pour", "dans",
        "que", "qui", "quoi", "voici", "voilà",
        "très", "bien", "mal", "ici", "là",
    ]
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
    if ext in ("txt", "md", "csv", "json", "xml", "html", "py", "js"):
        try:
            return file_bytes.decode("utf-8", errors="replace")[:15000]
        except Exception as e:
            return f"Erreur lecture texte: {e}"
    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = "".join(page.extract_text() or "" for page in reader.pages[:20])
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
    if ext in ("xlsx", "xls"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
            text = ""
            for sheet in wb.sheetnames[:3]:
                ws = wb[sheet]
                text += f"\n--- {sheet} ---\n"
                for row in ws.iter_rows(max_row=100, values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip():
                        text += row_text + "\n"
            return text[:15000]
        except Exception as e:
            return f"Erreur XLSX: {e}"
    return f"❌ Format '{ext}' non supporté."


def analyze_document(doc_text, filename, question, lang):
    system = (
        "Tu es Yelen AI, expert en analyse de documents. "
        "Analyse le document et réponds précisément. Sois concis et clair. "
        + ("Réponds en français ou wolof." if lang == "wolof" else "Réponds en français.")
    )
    q = question.strip() if question.strip() else "Fais un résumé complet de ce document."
    prompt = f'Document "{filename}" :\n\n---\n{doc_text[:12000]}\n---\n\nQuestion : {q}'
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=1000,
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"❌ Erreur analyse : {str(e)}"


# ─────────────────────────────────────────────────────────────
# TRADUCTION PROMPT IMAGE → ANGLAIS AVEC TEXTE FORCÉ
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str, text_to_write: str = None) -> str:
    """Traduit le prompt en anglais pour la génération d'image.
    Si text_to_write est fourni, on demande explicitement au traducteur
    de l'inclure tel quel dans le prompt traduit."""

    system = (
        "You are an image generation prompt translator and optimizer. "
        "Translate the user's request (in any language) into a detailed English image generation prompt. "
        "Return ONLY the English prompt, no explanation, no quotes."
    )

    user_content = prompt
    if text_to_write:
        user_content += (
            f"\n\nIMPORTANT: The image MUST include the exact text \"{text_to_write}\" "
            f"written clearly and legibly on it. Make sure the prompt explicitly asks for "
            f"this text to appear in the image."
        )

    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
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
# GÉNÉRATION IMAGE — TEXTE FORCÉ DANS LE PROMPT
# ─────────────────────────────────────────────────────────────
def _looks_like_image(raw: bytes) -> bool:
    if not raw or len(raw) < 500:
        return False
    if raw[:8] == b'\x89PNG\r\n\x1a\n': return True
    if raw[:2] == b'\xff\xd8':          return True
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP': return True
    if raw[:6] in (b'GIF87a', b'GIF89a'): return True
    return False


def generate_image(prompt: str, gen_type: str, text_to_write: str = None) -> str | None:
    english  = translate_prompt_to_english(prompt, text_to_write)
    prefix   = TYPE_PROMPTS.get(gen_type, "")
    w, h     = TYPE_SIZES.get(gen_type, (1024, 1024))

    # Construction du prompt final avec instructions de texte très explicites
    if text_to_write:
        text_instruction = (
            f'with the exact text "{text_to_write}" written in large bold clear readable letters, '
            f'text perfectly integrated into the design, high contrast legible typography, '
            f'the text "{text_to_write}" must be clearly visible and correctly spelled'
        )
        full = f"{prefix}, {english}, {text_instruction}, masterpiece, best quality, ultra detailed, sharp"
    else:
        full = f"{prefix}, {english}, masterpiece, best quality, ultra detailed, sharp"

    print(f"[IMAGE] type={gen_type} text='{text_to_write}' | {full[:100]}...")

    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt": full,
                    "width": min(w, 1024), "height": min(h, 1024),
                    "steps": 4, "n": 1, "response_format": "b64_json",
                },
                timeout=90,
            )
            r.raise_for_status()
            b64 = r.json()["data"][0]["b64_json"]
            if b64 and _looks_like_image(base64.b64decode(b64)):
                return b64
            print("[FLUX ERR] contenu non-image")
        except Exception as e:
            print("[FLUX ERR]", e)

    # Fallback Pollinations
    try:
        enc = urllib.parse.quote(full)
        url = f"https://image.pollinations.ai/prompt/{enc}?width={w}&height={h}&nologo=true&enhance=true&model=flux"
        res = requests.get(url, timeout=90)
        if res.status_code == 200 and _looks_like_image(res.content):
            return base64.b64encode(res.content).decode()
        print(f"[POLLINATIONS ERR] status={res.status_code}")
        return None
    except Exception as e:
        print("[POLLINATIONS ERR]", e)
        return None


# ─────────────────────────────────────────────────────────────
# ÉDITION IMAGE
# ─────────────────────────────────────────────────────────────
IMAGE_EDIT_KEYWORDS = [
    "modifie", "modifier", "change", "changer", "transforme",
    "édite", "editer", "remplace", "remplacer",
    "ajoute", "ajouter", "enlève", "enleve", "retire", "retirer",
    "supprime", "supprimer", "efface", "effacer",
    "colore", "colorie", "améliore", "ameliore",
    "soppi", "soppil",
]

def detect_image_edit_intent(msg: str) -> bool:
    m = _norm(msg)
    return any(_norm(kw) in m for kw in IMAGE_EDIT_KEYWORDS)


def edit_image(image_base64: str, mime: str, instruction: str, text_to_write: str = None) -> str | None:
    if not TOGETHER_API_KEY:
        return None
    english = translate_prompt_to_english(instruction, text_to_write)
    data_uri = f"data:{mime};base64,{image_base64}"
    try:
        r = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "black-forest-labs/FLUX.1-kontext-pro",
                "prompt": english,
                "image_url": data_uri,
                "width": 1024, "height": 1024,
                "steps": 28, "n": 1, "response_format": "b64_json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()["data"][0]
        b64 = data.get("b64_json")
        if b64 and _looks_like_image(base64.b64decode(b64)):
            return b64
        img_url = data.get("url")
        if img_url:
            img_res = requests.get(img_url, timeout=60)
            if img_res.status_code == 200 and _looks_like_image(img_res.content):
                return base64.b64encode(img_res.content).decode()
        return None
    except Exception as e:
        print("[EDIT IMG ERR]", e)
        return None


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
# TRANSCRIPTION AUDIO
# ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes) -> str:
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
        elif audio_bytes[:4] == b'\x1A\x45\xDF\xA3':
            suffix, mime = ".webm", "audio/webm"

    print(f"[AUDIO] {suffix} — {len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    # Phrase-exemple naturelle mêlant wolof et français —
    # biaise Whisper vers la bonne orthographe sans liste de mots isolés
    WHISPER_WOLOF_PROMPT = (
        "Nanga def? Jërejëf waay, mangi fi rekk, dama bëgg wax ak yow ci "
        "sama liggéey bi. Dafa neex na lool, waaye dama xam ne dinaa dem "
        "ci kër gi tey. Sama xarit yi ñëw ci dëkk bi, ñu ngi wax ci wolof "
        "ak français ñaari yoon. Bul fekk, dégg naa la bu baax, waaw waaw "
        "loolu moy dëgg. Yaangi dem fan? Mangi nekk ci internet bi. Baal "
        "ma, ndax dama bëgg nga jàppale ma ci nataal bi, defal ma ko su la "
        "neexee. Asalaa maalekum, maalekum salaam, jërejëf lool sama mbokk."
    )

    def _run(language):
        kwargs = dict(
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,
            prompt=WHISPER_WOLOF_PROMPT,
        )
        if language:
            kwargs["language"] = language
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(f"audio{suffix}", f, mime), **kwargs
            )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return (text or "").strip()

    try:
        candidates = []
        for lang in ("fr", None):
            try:
                text = _run(lang)
                if text:
                    candidates.append(text)
            except Exception as e:
                print(f"[WHISPER lang={lang} ERR]", e)

        if not candidates:
            return ""
        best = max(candidates, key=_wolof_score) if len(candidates) > 1 else candidates[0]
        print(f"[WHISPER] candidats={candidates!r} → retenu={best!r}")
        return best
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────
# CORRECTION WOLOF POST-TRANSCRIPTION
# ─────────────────────────────────────────────────────────────
_WOLOF_REFERENCE_TERMS = sorted(
    {w for w, score in WOLOF_WORDS.items() if score >= 3},
    key=len, reverse=True,
)[:120]

def correct_wolof_transcription(text: str) -> str:
    glossary = ", ".join(_WOLOF_REFERENCE_TERMS)
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un correcteur expert de transcriptions vocales wolof/français. "
                        "Whisper ne connaît pas le wolof et retranscrit souvent les mots wolof "
                        "par leur approximation phonétique en orthographe française.\n\n"
                        "Règles STRICTES :\n"
                        "- Ne traduis JAMAIS le wolof en français, ni l'inverse.\n"
                        "- Ne résume pas, ne reformule pas, ne complète pas.\n"
                        "- Corrige UNIQUEMENT l'orthographe des mots mal transcrits.\n"
                        "- Garde le mélange wolof-français si c'est le cas.\n"
                        "- Déformations fréquentes : 'nan ga def' → 'nanga def', "
                        "'jere jef' → 'jërejëf', 'deug' → 'dëgg', 'beug' → 'bëgg', "
                        "'cham'/'kham' → 'xam', 'gnew' → 'ñëw', 'wow' → 'waaw', "
                        "'didi' → 'deedeet', 'rekke' → 'rekk', 'sama xarite' → 'sama xarit'.\n"
                        f"- Vocabulaire de référence : {glossary}.\n\n"
                        "Retourne UNIQUEMENT la transcription corrigée, rien d'autre."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=500,
        )
        corrected = r.choices[0].message.content.strip()
        print(f"[CORRECTION] '{text}' → '{corrected}'")
        return corrected
    except Exception as e:
        print("[CORRECTION ERR]", e)
        return text


# ─────────────────────────────────────────────────────────────
# SYNTHÈSE VOCALE — gTTS + pré-traitement wolof
# ─────────────────────────────────────────────────────────────
# Le wolof n'est pas supporté par gTTS. On utilise le français
# comme approximation et on pré-traite le texte pour améliorer
# la prononciation : translittération des diacritiques wolof
# vers leurs équivalents phonétiques français, et remplacement
# des mots wolof courants par leur prononciation approximative.

_WOLOF_PRONUNCIATION = {
    # Diacritiques wolof → phonétique française
    "ë": "é", "ñ": "gn", "ŋ": "ng",
    # Mots wolof courants → comment les lire en français
    "jërejëf": "djérédjef", "waaw": "waw", "deedeet": "dédet",
    "mangi fi": "man-gui fi", "maa ngi": "ma ngui",
    "dafa": "dafa", "dama": "dama", "bëgg": "bèg",
    "rekk": "rèk", "lool": "lol", "ndax": "ndak",
    "xam": "kham", "ñëw": "gnew", "ñu": "gnu",
    "nekk": "nèk", "dem": "dèm", "lekk": "lèk",
    "wax": "wak", "gis": "guiss", "jox": "djok",
    "teg": "tèg", "fal": "fal", "muñ": "mougn",
    "soppi": "soppi", "xool": "khol",
    "kaay": "kay", "daldi": "daldi",
    "sama": "sama", "seen": "sène", "suñu": "sougnou",
    "baax": "bak", "rafet": "rafète", "neex": "nèkh",
    "yëgël": "yéguèl", "baal ma": "bal ma",
    "asalaa maalekum": "assalamu alaykoum",
    "maalekum salaam": "alaykoumoussalam",
    "incha allah": "inchallah", "yalla": "yalla",
    "baraka": "baraka",
}

_MD_CLEAN_RE    = re.compile(r'[*_`#>~\[\]()\|]')
_EMOJI_CLEAN_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+"
)

def _preprocess_for_tts(text: str, lang_detected: str) -> str:
    """Nettoie le texte et, si du wolof est détecté, translittère
    les mots wolof pour une meilleure prononciation en gTTS français."""
    # Nettoyage général
    t = _EMOJI_CLEAN_RE.sub("", text)
    t = _MD_CLEAN_RE.sub("", t)
    t = re.sub(r'\s+', ' ', t).strip()

    if lang_detected == "wolof":
        # Remplace les mots wolof par leurs approximations phonétiques
        for wolof_word, french_pronounce in sorted(
            _WOLOF_PRONUNCIATION.items(), key=lambda x: -len(x[0])
        ):
            t = re.sub(
                re.escape(wolof_word),
                french_pronounce,
                t,
                flags=re.IGNORECASE
            )

    return t[:1200]


def generate_speech(text: str, lang_detected: str = "french") -> str | None:
    if not TTS_AVAILABLE:
        return None

    clean = _preprocess_for_tts(text, lang_detected)
    if not clean:
        return None

    try:
        buf = io.BytesIO()
        # On utilise toujours le français (meilleure approximation pour le wolof)
        gTTS(text=clean, lang="fr", slow=False).write_to_fp(buf)
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        print(f"[TTS] généré {len(buf.getvalue())} bytes")
        return audio_b64
    except Exception as e:
        print("[TTS ERR]", e)
        return None


# ─────────────────────────────────────────────────────────────
# SYSTÈMES DE PROMPTS
# ─────────────────────────────────────────────────────────────
WOLOF_SYSTEM = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles couramment le wolof et le français, comme un jeune Dakarois éduqué des années 2020.

RÈGLES DE LANGUE :
1. MESSAGE EN WOLOF PUR → réponds EN WOLOF.
2. MESSAGE MÉLANGÉ wolof-français → réponds dans le MÊME style mélangé.
3. MESSAGE EN FRANÇAIS PUR → réponds EN FRANÇAIS.
4. Sois NATUREL, CONCIS, CHALEUREUX.
5. N'invente JAMAIS de mots wolof inexistants.
6. Si tu ne comprends pas, demande à l'interlocuteur de répéter.

GRAMMAIRE ESSENTIELLE :
• Dama + verbe = Je... | Dafa + verbe = Il/Elle... | Mangi = Je suis en train de...
• Négation : -ul (suffixe) | duma (je ne) | dul (il ne)
• Questions : Naka? (Comment) | Lan? (Quoi) | Fan? (Où) | Ndax? (Est-ce que)

EXEMPLES :
- "Nanga def?" → "Mangi fi rekk, jërejëf! Yow noo?"
- "Dama bëgg nataal" → "Waaw! Xam ma lan nga bëgg — logo, illustration, walla yenn?"
- "Logo bi rafet na!" → "Jërejëf lool! Danga bëgg yenn add?"
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
            confirmation = intent["confirmation_message"]
            if text_to_write:
                confirmation = f"🎨 Image générée avec le texte « {text_to_write} » !"
            return {
                "response":     confirmation,
                "has_image":    True,
                "image_base64": img,
                "image_type":   intent["type"],
                "visual_prompt": intent["visual_prompt"],
                "text_written": text_to_write,
            }
        return {"response": "❌ Génération échouée. Réessaie dans quelques secondes."}

    lang   = detect_language(user_message)
    system = WOLOF_SYSTEM if lang == "wolof" else FRENCH_SYSTEM
    print(f"[LANG DETECTED] {lang} | message='{user_message[:60]}'")

    messages = [{"role": "system", "content": system}]
    for msg in history[-10:]:
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=600,
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
    if not data:
        return jsonify({"error": "invalid request"}), 400
    text = data.get("text", "").strip()
    lang = data.get("lang", "french")
    if not text:
        return jsonify({"error": "Texte vide"}), 400
    audio = generate_speech(text, lang)
    if not audio:
        return jsonify({"error": "Synthèse vocale indisponible"}), 500
    return jsonify({"audio_base64": audio, "has_audio_response": True})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid request"}), 400

    user_message = data.get("message", "")
    has_image    = data.get("has_image", False)
    image_base64 = data.get("image_base64")
    has_audio    = data.get("has_audio", False)
    audio_base64 = data.get("audio_base64")
    has_document = data.get("has_document", False)
    doc_base64   = data.get("doc_base64")
    doc_filename = data.get("doc_filename", "document.txt")
    history      = data.get("history", [])

    # ── 🎙 AUDIO ──
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)
            if len(audio_bytes) < 500:
                return jsonify({"response": "❌ Audio trop court."})

            transcribed = transcribe_audio(audio_bytes)
            transcribed = correct_wolof_transcription(transcribed)
            print("[TRANSCRIPTION FINALE]", repr(transcribed))

            if not transcribed or len(transcribed.strip()) < 2:
                return jsonify({"response": "❌ Audio non reconnu. Rapproche-toi du micro."})

            try:
                result = handle_chat(transcribed, history)
            except Exception as e:
                result = {"response": f"❌ Erreur lors de la réponse : {str(e)}"}

            lang_detected = result.pop("_lang", detect_language(transcribed))
            result["transcription"]    = transcribed
            result["is_voice_message"] = True

            # TTS automatique quand l'user parle (vocal → vocal)
            response_text = result.get("response", "")
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
            if doc_text.startswith("❌"):
                return jsonify({"response": doc_text})
            question = user_message.strip()
            lang     = detect_language(question) if question else "french"
            response = analyze_document(doc_text, doc_filename, question, lang)
            return jsonify({"response": response, "has_document": True, "doc_filename": doc_filename})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur document : {str(e)}"})

    # ── 🖼 IMAGE : ÉDITION ou ANALYSE ──
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})
            mime     = get_mime(image_base64)
            question = user_message.strip()

            if question and detect_image_edit_intent(question):
                text_to_write = extract_text_to_write(question)
                edited = edit_image(image_base64, mime, question, text_to_write)
                if edited:
                    msg = "🎨 Voici l'image modifiée !"
                    if text_to_write:
                        msg = f"🎨 Image modifiée avec le texte « {text_to_write} » !"
                    return jsonify({"response": msg, "has_image": True, "image_base64": edited, "image_type": "edit"})
                return jsonify({"response": "❌ Modification échouée. Réessaie."})

            question = question or "Décris cette image en détail en français."
            for model in ["meta-llama/llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-maverick-17b-128e-instruct"]:
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
