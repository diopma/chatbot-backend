import os
import base64
import tempfile
import urllib.parse
import requests
import io

from flask import Flask, request, jsonify
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
# DÉTECTION LANGUE — vocabulaire wolof étendu et pondéré
# ─────────────────────────────────────────────────────────────
WOLOF_WORDS = {

    # ══ Score 5 : expressions uniquement wolof, très distinctives ══
    "nanga def":        5, "nanga xam":       5, "nanga dem":       5,
    "mangi fi rekk":    5, "maa ngi fi":      5, "mangi dem":       5,
    "jërejëf lool":     5, "baal ma ko":      5, "waaw waaw":       5,
    "dafa baax na":     5, "dafa neex na":    5, "dafa rafet na":   5,
    "dafa mel ni":      5, "xam naa":         5, "faamaak":         5,
    "amul solo":        5, "def ma":          5, "bind ma":         5,
    "yëgël ma":         5, "soo bëgg":        5, "bëgg naa":        5,
    "lu baax":          5, "lu neex":         5, "maa ngi dem":     5,
    "asalaa maalekum":  5, "maalekum salaam": 5,
    "dama bëgg":        5, "dama dem":        5, "dama nekk":       5,
    "yow noo":          5, "lan moy":         5, "fan nga dem":     5,
    "naka nga def":     5, "naka waay":       5,
    "bul fekk":         5, "bul ko wax":      5,
    "dëgg naa":         5, "dégg naa":        5,
    "fii rekk":         5, "fi rekk":         5,
    "mooy":             5, "mooye":           5,
    "loolu moy":        5, "lool la":         5,
    "waxoon na":        5, "waxoon naa":      5,
    "liggéeyal ma":     5, "liggéeyal":       5,
    "sëriñ":            5,

    # ══ Score 4 : mots très distinctifs wolof ══
    "jërejëf":  4, "jërëjëf":  4, "baal ma":   4,
    "deedeet":  4, "mangi fi":  4, "mangi":     4,
    "maa ngi":  4, "nataal":    4, "nataalu":   4,
    "liggéey":  4, "liggeeyu":  4,
    "dafa":     4, "dama":      4, "xam":       4,
    "bëgg":     4, "siiw":      4, "rekk":      4,
    "yëgël":    4, "woneel":    4, "wone":      4,
    "ndanka":   4, "ndanka ndanka": 4,
    "mbokk":    4, "xarit":     4,
    "télé bi":  4, "kër gi":    4, "dekk bi":   4,
    "suñu":     4, "sunu":      4,
    "dëkk":     4, "xeex":      4,
    "tubaab":   4, "nit":       4, "niit":      4,
    "wolof":    4, "seereer":   4, "pulaar":    4,
    "muñ":      4, "muñël":     4,
    "tëgg":     4, "tëgël":     4,
    "fëkk":     4, "fëkkeel":   4,
    "digg":     4, "digël":     4,
    "taaw":     4, "taawël":    4,
    "soppi":    4, "soppil":    4,
    "añ":       4, "añël":      4,
    "jaaxle":   4, "jaaxal":    4,
    "ñëlëm":    4, "xool":      4, "xoolël":    4,
    "wëñ":      4, "wëñël":     4,
    "bëgg bëgg": 4,
    "ñaar":     4, "ñett":      4, "ñent":      4,
    "juróom":   4, "fukk":      4, "téeméer":   4,
    "junni":    4,
    "ñaar fukk": 4, "ñett fukk": 4,

    # ══ Score 3 : courants en wolof ══
    "waaw":   3, "yow":    3, "moom":    3,
    "laa":    3, "naa":    3, "nga":     3,
    "nekk":   3, "topp":   3, "wax":     3,
    "gis":    3, "gisul":  3,
    "rafet":  3, "baax":   3, "neex":    3,
    "ndax":   3, "waaye":  3, "mbaa":    3,
    "tey":    3, "bëccëk": 3, "guddi":   3,
    "jaay":   3, "jënd":   3,
    "xol":    3, "bàkkaar": 3,
    "daldi":  3, "seet":   3,
    "leen":   3, "sunu":   3,
    "fanaan": 3, "tëëy":   3,
    "kaay":   3, "kaaye":  3,
    "ñëw":    3, "dem":    3,
    "lekk":   3, "dox":    3,
    "bind":   3, "jëf":    3,
    "tëdd":   3, "xaar":   3,
    "jox":    3, "joxël":  3,
    "naan":   3, "naanël":  3,
    "xëy":    3, "xëyël":  3,
    "teg":    3, "tegël":  3,
    "fal":    3, "falël":  3,
    "gën":    3, "gëna":   3,
    "sol":    3, "solël":  3,
    "door":   3, "doorël": 3,

    # ══ Score 2 : particules, pronoms, articles wolof ══
    "ak":   2, "sama":  2, "seen":  2,
    "ci":   2, "bi":    2, "yi":    2,
    "bu":   2, "si":    2, "gi":    2,
    "ki":   2, "ji":    2, "li":    2,
    "baay": 2, "yaay":  2, "xale":  2,
    "goor": 2, "jigéen": 2, "doom": 2,
    "benn": 2, "lool":  2, "def":   2,
    "am":   2, "ko":    2, "len":   2,
    "mu":   2, "nu":    2, "ñu":    2,
    "set":  2, "setal": 2, "tax":   2,
    "di":   2, "la":    2, "le":    2,
    "na":   2, "ni":    2, "fi":    2,
    "ba":   2, "ca":    2, "ga":    2,
    "mo":   2, "ñi":    2,

    # ══ Expressions nouchi dakarois (mélange wolof-français) ══
    "dafa nice":    3, "dafa good":    3, "dafa classe":   3,
    "waaw frère":   3, "waaw sama xarit": 3,
    "bonne journée bi": 3, "bonne nuit bi": 3,
    "incha allah":  2, "alhamdoulilah": 2,
    "bismillah":    2, "masha allah":   2,
    "yalla":        3, "yalla boole":   3,
    "baraka":       3, "baraka allah":  3,
}

def detect_language(text: str) -> str:
    t = text.lower()

    wolof_score = 0
    for word, weight in WOLOF_WORDS.items():
        if word in t:
            wolof_score += weight

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

    if wolof_score >= 3:                          return "wolof"
    if wolof_score >= 2 and french_score <= 3:    return "wolof"
    if wolof_score >= 1 and french_score >= 1:    return "wolof"
    if french_score >= 3:                         return "french"
    return "french"


# ─────────────────────────────────────────────────────────────
# EXTRACTION TEXTE DOCUMENT
# ─────────────────────────────────────────────────────────────
def extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    print(f"[DOC] Extension: {ext} | Taille: {len(file_bytes)} bytes")

    if ext in ("txt", "md", "csv", "json", "xml", "html", "py", "js"):
        try:
            return file_bytes.decode("utf-8", errors="replace")[:15000]
        except Exception as e:
            return f"Erreur lecture texte: {e}"

    if ext == "pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages[:20]:
                text += page.extract_text() or ""
            return text[:15000] if text.strip() else "PDF sans texte extractible (probablement scanné)."
        except ImportError:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                text = ""
                for page in reader.pages[:20]:
                    text += page.extract_text() or ""
                return text[:15000]
            except Exception as e:
                return f"Erreur lecture PDF: {e}"
        except Exception as e:
            return f"Erreur PDF: {e}"

    if ext == "docx":
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text[:15000]
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
                    row_text = " | ".join([str(c) for c in row if c is not None])
                    if row_text.strip():
                        text += row_text + "\n"
            return text[:15000]
        except ImportError:
            return "❌ openpyxl non installé."
        except Exception as e:
            return f"Erreur XLSX: {e}"

    return f"❌ Format '{ext}' non supporté. Formats acceptés : PDF, DOCX, TXT, CSV, XLSX, MD."


# ─────────────────────────────────────────────────────────────
# ANALYSE DOCUMENT VIA LLM
# ─────────────────────────────────────────────────────────────
def analyze_document(doc_text: str, filename: str, question: str, lang: str) -> str:
    if lang == "wolof":
        system = (
            "Tu es Yelen AI, assistant sénégalais expert en analyse de documents. "
            "Réponds en français ou wolof selon la langue de la question. "
            "Analyse le document fourni et réponds précisément à la question posée. "
            "Sois concis et clair."
        )
    else:
        system = (
            "Tu es Yelen AI, un assistant expert en analyse de documents. "
            "Analyse le document fourni et réponds précisément à la question posée. "
            "Sois concis, structuré et clair. Réponds en français."
        )

    q = question.strip() if question.strip() else "Fais un résumé complet de ce document."
    prompt = f'Voici le contenu du document "{filename}" :\n\n---\n{doc_text[:12000]}\n---\n\nQuestion : {q}'

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"[DOC LLM ERR] {e}")
        return f"❌ Erreur lors de l'analyse : {str(e)}"


# ─────────────────────────────────────────────────────────────
# TRADUCTION PROMPT → ANGLAIS
# ─────────────────────────────────────────────────────────────
def translate_prompt_to_english(prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an image prompt translator. Translate the user's request (in any language) into a detailed English image generation prompt. Return ONLY the English prompt, no explanation, no quotes."},
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
    print(f"[IMAGE] type={gen_type} | {full[:80]}...")

    if TOGETHER_API_KEY:
        try:
            r = requests.post(
                "https://api.together.xyz/v1/images/generations",
                headers={"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"},
                json={"model": "black-forest-labs/FLUX.1-schnell-Free", "prompt": full, "width": min(w,1024), "height": min(h,1024), "steps": 4, "n": 1, "response_format": "b64_json"},
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

    print(f"[AUDIO] {suffix} — {len(audio_bytes)} bytes")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name

    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=(f"audio{suffix}", f, mime),
                response_format="text",
                prompt=(
                    "Transcris exactement ce qui est dit en wolof sénégalais ou français. "
                    "Le wolof a ces caractéristiques : ë (e avec tréma), ñ, ŋ, accent tonique. "
                    "Mots wolof très fréquents : "
                    "nanga def, mangi fi rekk, jërejëf, waaw, deedeet, dama, dafa, xam, "
                    "dem, ñëw, lekk, wax, nekk, bi, yi, ci, ak, sama, bëgg, nataal, "
                    "def ma, bind ma, xale, baay, yaay, xarit, mbokk, ndax, baal ma, "
                    "yow, moom, sunu, seen, rekk, lool, daldi, kaay, ñu, mu, "
                    "gis, jox, naan, teg, fal, sol, door, xool, soppi, muñ, "
                    "tubaab, wolof, seereer, kër, suñu, yalla, baraka, incha allah, "
                    "asalaa maalekum, maalekum salaam. "
                    "Termes tech possibles : logo, image, avatar, créer, générer, intelligence artificielle."
                ),
            )
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        print(f"[WHISPER] texte={repr(text)}")
        return (text or "").strip()
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────
# SYSTÈME DE PROMPTS — WOLOF AMÉLIORÉ
# ─────────────────────────────────────────────────────────────
WOLOF_SYSTEM = """Tu es Yelen AI, un assistant IA sénégalais intelligent, chaleureux et moderne.
Tu parles couramment le wolof et le français, comme un jeune Dakarois éduqué des années 2020.

═══════════════════════════════════════════════
RÈGLES DE LANGUE
═══════════════════════════════════════════════

1. MESSAGE EN WOLOF PUR → réponds EN WOLOF (avec du français technique si besoin).
2. MESSAGE MÉLANGÉ wolof-français (nouchi dakarois) → réponds dans le MÊME style mélangé.
3. MESSAGE EN FRANÇAIS PUR → réponds EN FRANÇAIS.
4. Sois NATUREL, CONCIS, CHALEUREUX. Pas de réponses trop longues.
5. N'invente JAMAIS de mots wolof inexistants. Si tu ne sais pas un mot, utilise le français.
6. Utilise des interjections naturelles comme : "Waaw !", "Dafa baax !", "Yëgël na !", "Jërejëf !"

═══════════════════════════════════════════════
GRAMMAIRE WOLOF ESSENTIELLE
═══════════════════════════════════════════════

STRUCTURE DE BASE :
• Sujet + Prédicat verbal + Objet
• Le verbe s'accorde avec le focus (sujet ou verbe mis en avant)

CONJUGAISON PRÉSENT :
• Dama + verbe = Je + verbe (focus sujet)    Ex: "Dama dem" = Je vais
• Dafa + verbe = Il/Elle + verbe             Ex: "Dafa lekk" = Il mange
• Danga + verbe = Tu + verbe                 Ex: "Danga xam" = Tu sais
• Mangi + verbe = Je suis en train de        Ex: "Mangi liggéey" = Je travaille
• Maa ngi + verbe = Je suis (état présent)   Ex: "Maa ngi fi" = Je suis ici

NÉGATION :
• doo → tu ne... pas     Ex: "Doo dem" = Tu ne vas pas
• duma → je ne... pas    Ex: "Duma dem" = Je ne vais pas
•dul → il ne... pas     Ex: "Dul lekk" = Il ne mange pas
• -ul (suffixe)          Ex: "Xamul" = Il ne sait pas, "Bëggul" = Il ne veut pas

QUESTIONS :
• "Naka...?" = Comment...?      Ex: "Naka nga def?" = Comment tu vas?
• "Lan...?" = Quoi/Que...?      Ex: "Lan la?" = C'est quoi?
• "Fan...?" = Où...?            Ex: "Fan nga dem?" = Tu vas où?
• "Kan...?" = Qui...?           Ex: "Kan la wax?" = Qui t'a dit?
• "Ndax...?" = Est-ce que...?   Ex: "Ndax danga dem?" = Tu vas?
• "Buki...?" = Combien...?      Ex: "Buki?" = Combien ça coûte?
• "Kañ...?" = Quand...?         Ex: "Kañ nga ñëw?" = Tu viens quand?

ARTICLES DÉFINIS (dépendent de la classe nominale) :
• bi = le/la (objets singuliers, personnes)  Ex: "xale bi" = l'enfant
• gi = le/la (lieux, certains noms)          Ex: "kër gi" = la maison
• ji = le/la (corps, certains noms)          Ex: "bopp ji" = la tête
• yi = les (pluriel)                         Ex: "xale yi" = les enfants
• si = le/la (certains objets)               Ex: "bis si" = le bus
• ki = cette personne-ci                     Ex: "xale ki" = cet enfant-ci

POSSESSION :
• sama = mon/ma/mes           Ex: "sama xarit" = mon ami
• sa = ton/ta/tes             Ex: "sa kër" = ta maison
• mu = son/sa (3e pers sing)  Ex: "mu dem" = il/elle est parti(e)
• sunu = notre                Ex: "sunu reew" = notre pays
• seen = leur/leurs           Ex: "seen kër" = leur maison

═══════════════════════════════════════════════
VOCABULAIRE COMPLET PAR THÈME
═══════════════════════════════════════════════

── SALUTATIONS ──
• "Nanga def?" / "Naka nga def?" → Comment vas-tu?
• "Mangi fi rekk, jërejëf!" → Je vais bien, merci!
• "Asalaa maalekum" → Paix sur vous (salut islamique)
• "Maalekum salaam" → Et sur vous la paix
• "Mbaa dëkk?" → Comment ça va? (informel)
• "Yow noo?" → Et toi, comment tu vas?
• "Dafa baax" → Ça va bien / C'est bien
• "Fanaan?" → Tu as bien dormi?
• "Naka suba si?" → Comment va ce matin?

── REMERCIEMENTS / POLITESSE ──
• "Jërejëf" → Merci
• "Jërejëf lool" → Merci beaucoup
• "Baal ma" → Excuse-moi / Pardon
• "Baal ma ko" → Excuse-moi pour ça
• "Amul solo" → Pas de problème / De rien
• "Yëgël na!" → Je comprends! / Bien sûr!
• "Waaw, yëgël naa" → Oui, je comprends
• "Dëgg naa" → J'ai entendu / J'ai compris

── OUI / NON / CONFIRMATION ──
• "Waaw" → Oui
• "Waaw waaw" → Oui oui (confirmer fort)
• "Deedeet" → Non
• "Siiw" / "Siiw rekk" → Exactement / C'est ça
• "Dëgg" → Vrai / Exact
• "Mën na" → C'est possible / Oui ça peut
• "Mënul" → Ce n'est pas possible / Non
• "Benn" → Non (fort refus)

── ÉMOTIONS ET ÉTATS ──
• "Dafa baax" / "Dafa baax na" → C'est bien
• "Dafa neex" / "Dafa neex na" → C'est bon / C'est agréable
• "Dafa rafet" / "Dafa rafet na" → C'est beau
• "Dafa mel ni..." → Ça ressemble à...
• "Neex na lool" → C'est vraiment bien
• "Rafet lool" → C'est très beau
• "Dafa xel" → C'est intelligent / Ça a l'air fort
• "Dafa gëna baax" → C'est encore mieux
• "Dafa metti" → C'est difficile / Ça fait mal
• "Dafa sedd" → C'est froid
• "Dafa tang" → C'est chaud
• "Dafa am solo" → C'est important
• "Amul solo" → Ça n'a pas d'importance
• "Dafa xam" → Il/elle sait
• "Xamul" → Il/elle ne sait pas

── VERBES COURANTS ──
• dem → aller              • ñëw → venir
• lekk → manger            • naan → boire
• wax → parler/dire        • xam → savoir/connaître
• bëgg → vouloir/aimer     • gis → voir
• jëf → faire (action)     • def → faire/créer
• bind → écrire/dessiner   • dox → marcher/aller
• jaay → vendre            • jënd → acheter
• topp → suivre            • seet → regarder/chercher
• nekk → être/se trouver   • am → avoir
• fëkk → trouver           • tëdd → dormir/se coucher
• xaar → attendre          • jox → donner
• teg → mettre/poser       • fal → choisir
• gën → dépasser/mieux     • muñ → supporter/patienter
• soppi → changer          • añ → déjeuner
• sol → porter (vêtement)  • door → commencer
• xool → regarder/observer • naan → boire
• liggéey → travailler     • xëy → aimer (quelqu'un fort)
• bañ → refuser/ne pas vouloir • mën → pouvoir
• wóor → être clair/évident • taxaw → s'arrêter/debout

── PRONOMS ET PARTICULES ──
• Man/Maa → Moi/Je          • Yow → Toi
• Moom → Lui/Elle           • Nun/Nunu → Nous
• Yéen → Vous               • Ñoom → Eux/Elles
• Sama → Mon/Ma             • Sa → Ton/Ta
• Bu/Bi/Gi → article défini • Yi → Les (pluriel)
• Ci → à/dans/sur/y         • Ak → et/avec
• Rekk → seulement/juste    • Lool → vraiment/beaucoup
• Doon → était/avant (passé) • Dul → ne...pas (3e pers)

── TEMPS ET MOMENTS ──
• Tey → Aujourd'hui          • Démb → Hier
• Elëgg → Demain             • Bëccëk → Ce matin
• Guddi → Ce soir/La nuit    • Kere/Kerë → Midi/tantôt
• Xawaré → L'après-midi      • Fanaan → Cette nuit passée
• Ñaari fan → Dans deux jours • Bi tey → Aujourd'hui même

── FAMILLE ──
• Baay → Papa               • Yaay → Maman
• Xale → Enfant             • Doom → Enfant (de qqn)
• Mbeebar → Petit-enfant    • Mag → Aîné/grand(e)
• Rakk → Cadet/cadette      • Jëkkër → Mari
• Jabar → Femme (épouse)    • Mbokk → Parent/proche
• Xarit → Ami(e)            • Goro → Camarade/pote

── LIEUX ──
• Kër → Maison              • Dëkk → Village/ville/quartier
• Yëgël → Route (vers)      • Fii → Ici
• Fale → Là-bas             • Fi ci kanam → Devant
• Fi ci ginnaaw → Derrière  • Suuf bi → La terre/le sol
• Géej bi → La mer          • Réew mi → Le pays

── NOMBRES ──
• Benn → 1    • Ñaar → 2    • Ñett → 3
• Ñent → 4    • Juróom → 5  • Juróom benn → 6
• Juróom ñaar → 7  • Juróom ñett → 8  • Juróom ñent → 9
• Fukk → 10   • Fukk ak benn → 11  • Ñaar fukk → 20
• Téeméer → 100  • Junni → 1000

── EXPRESSIONS POPULAIRES DAKAR ──
• "Kaay fi!" → Viens ici!
• "Daldi dem!" → Vite, vas-y!
• "Ndanka ndanka" → Doucement / Petit à petit
• "Waaw rekk!" → Exactement!
• "Maa ko def" → Je vais le faire / Je m'en occupe
• "Xam naa" → Je sais / Je comprends
• "Xamul" → Il/elle ne sait pas
• "Dëkk bi dafa xew" → La ville bouge / Il se passe des choses
• "Bu baax" → Si c'est bien
• "Soo bëgg" → Si tu veux
• "Jaraay na" → Il/elle s'est battu(e) / Il a essayé
• "Bul faj!" → Ne t'inquiète pas! (litt: ne te soigne pas)
• "Yaw la tax" → C'est à cause de toi
• "Nit ku baax" → Une bonne personne
• "Dafa am yaram" → Il/elle a du charisme / Il est classe
• "Xam sa bopp" → Connais-toi toi-même / Fais attention
• "Bëgg bëgg" → Vraiment vouloir / Adorer
• "Dafa gëna rafet" → C'est encore plus beau

── INTELLIGENCE ARTIFICIELLE EN WOLOF ──
• "Intelligence artificielle" → "Intelligence artificielle bi" (on garde le terme français)
• "Application bi" → L'application
• "Yelen AI mooy..." → Yelen AI c'est...
• "Maa ngi Yelen AI" → Je suis Yelen AI
• "Dama mën def image" → Je peux créer une image
• "Dama mën wax ak yow" → Je peux discuter avec toi
• "Jëmël ma sa laaj" → Pose-moi ta question

═══════════════════════════════════════════════
EXEMPLES DE CONVERSATIONS COMPLÈTES
═══════════════════════════════════════════════

Utilisateur: "Nanga def?"
Réponse: "Mangi fi rekk, jërejëf! Yow noo? yaw nak nakamou."

Utilisateur: "Dama bëgg xam intelligence artificielle bi"
Réponse: "Waaw, xam naa! Intelligence artificielle bi moy ordinateur bu mën waxtaan, def nataal, ak bind — nu mel ni nit. Dama jëf ci yow pour yëgël la ko ndax mu neex."

Utilisateur: "Def ma yenn nataal"
Réponse: "Waaw! Xam ma lan nga bëgg nataal bi — logo, illustration, walla yenn? Wax ma si ci laaj bi, dinaa jëf ko dëgg dëgg."

Utilisateur: "Logo bi rafet na!"
Réponse: "Jërejëf lool! Dafa rafet, dëgg la wax. Danga bëgg yenn add ci wow?"

Utilisateur: "Lan moy Yelen AI?"
Réponse: "Maa ngi Yelen AI — intelligence artificielle bu Sénégal. Dama wax wolof ak français, dama mën def nataal yi, ak dama mën jaglël la ci sa liggéey yi. Kan maa tax jëmël ma sa laaj!"

Utilisateur: "Yaangi dem fan?"
Réponse: "Mangi nekk ci internet bi, dégëlu kaw! Dama dem ci sa kow toujours."

Utilisateur: "Baal ma, duma xam wolof bu baax"
Réponse: "Amul solo! Wax français rekk bëgg bëgg, dinaa response ci français. Yelen AI dafa mën wax ñaari làkk — wolof ak français. Laaj lañ bëgg!"
"""

FRENCH_SYSTEM = (
    "Tu es Yelen AI, un assistant IA africain intelligent, chaleureux et concis. "
    "Tu réponds toujours en français. "
    "Tu peux créer des images, logos, illustrations, analyser des documents PDF/Word/Excel. "
    "Sois naturel, utile et positif."
)

# ─────────────────────────────────────────────────────────────
# HANDLE CHAT
# ─────────────────────────────────────────────────────────────
def handle_chat(user_message: str, history: list) -> dict:
    intent = detect_image_intent(user_message)
    if intent:
        img = generate_image(intent["visual_prompt"], intent["type"])
        if img:
            return {"response": intent["confirmation_message"], "has_image": True, "image_base64": img, "image_type": intent["type"], "visual_prompt": intent["visual_prompt"]}
        return {"response": "❌ Génération échouée. Réessaie dans quelques secondes."}

    lang = detect_language(user_message)
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
    return {"response": r.choices[0].message.content}

# ─────────────────────────────────────────────────────────────
# KEEP-ALIVE
# ─────────────────────────────────────────────────────────────
@app.route("/ping", methods=["GET"])
def ping():
    return "pong", 200

# ─────────────────────────────────────────────────────────────
# ROUTE /chat
# ─────────────────────────────────────────────────────────────
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
                return jsonify({"response": "❌ Audio trop court. Parle un peu plus longtemps."})
            transcribed = transcribe_audio(audio_bytes)
            print("[TRANSCRIPTION]", repr(transcribed))
            if not transcribed or len(transcribed.strip()) < 2:
                return jsonify({"response": "❌ Audio non reconnu. Rapproche-toi du micro et réessaie."})
            result = handle_chat(transcribed, history)
            result["transcription"]    = transcribed
            result["is_voice_message"] = True
            return jsonify(result)
        except Exception as e:
            print("[AUDIO ERR]", e)
            return jsonify({"response": f"❌ Erreur audio : {str(e)}"})

    # ── 📄 DOCUMENT ──
    if has_document and doc_base64:
        try:
            doc_bytes = base64.b64decode(doc_base64)
            print(f"[DOC] Fichier reçu: {doc_filename} | {len(doc_bytes)} bytes")
            doc_text = extract_text_from_document(doc_bytes, doc_filename)
            if doc_text.startswith("❌"):
                return jsonify({"response": doc_text})
            print(f"[DOC] Texte extrait: {len(doc_text)} caractères")
            question = user_message.strip()
            lang = detect_language(question) if question else "french"
            response = analyze_document(doc_text, doc_filename, question, lang)
            return jsonify({
                "response":     response,
                "has_document": True,
                "doc_filename": doc_filename,
                "doc_chars":    len(doc_text),
            })
        except Exception as e:
            print("[DOC ERR]", e)
            return jsonify({"response": f"❌ Erreur lecture document : {str(e)}"})

    # ── 🖼 ANALYSE IMAGE ──
    if has_image and image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            if len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide."})
            mime     = get_mime(image_base64)
            question = user_message.strip() or "Décris cette image en détail en français."
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
            return jsonify({"response": "❌ Analyse image impossible. Réessaie."})
        except Exception as e:
            return jsonify({"response": f"❌ Erreur : {str(e)}"})

    # ── 💬 TEXTE ──
    if not user_message.strip():
        return jsonify({"response": "❌ Message vide."})

    try:
        return jsonify(handle_chat(user_message, history))
    except Exception as e:
        print("[CHAT ERR]", e)
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Yelen AI API 🌟"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
