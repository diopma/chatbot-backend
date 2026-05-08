# ─────────────────────────────────────────────────────────────
# CORRECTION 1 (backend) : Ajouter un flag "force_analysis"
# pour éviter que detect_image_intent intercepte les requêtes
# d'analyse d'image envoyées depuis le frontend.
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
    history       = data.get("history", [])

    # ─────────────────────────────
    # 🎙 AUDIO → TRANSCRIPTION
    # ─────────────────────────────
    if has_audio and audio_base64:
        try:
            audio_bytes = base64.b64decode(audio_base64)

            # CORRECTION AUDIO : détecter le vrai format du fichier
            # Expo enregistre en m4a/aac sur iOS et 3gp/amr sur Android
            # Whisper accepte mp3, mp4, mpeg, mpga, m4a, wav, webm
            # On utilise .m4a comme extension par défaut (compatible iOS + Android)
            suffix = ".m4a"
            mime   = "audio/m4a"

            # Détection basique du format par magic bytes
            if audio_bytes[:4] == b'RIFF':
                suffix = ".wav"
                mime   = "audio/wav"
            elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
                suffix = ".mp3"
                mime   = "audio/mp3"
            elif audio_bytes[4:8] == b'ftyp':
                suffix = ".m4a"
                mime   = "audio/m4a"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                path = tmp.name

            with open(path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=(f"audio{suffix}", f, mime),
                    response_format="text",
                )
            os.unlink(path)

            user_message = transcription if isinstance(transcription, str) else transcription.text
            print("[TRANSCRIPTION]", repr(user_message))

            if not user_message or not user_message.strip():
                return jsonify({"response": "❌ Audio non reconnu. Parle plus clairement ou plus fort."})

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

            # Vérifier que le base64 n'est pas vide
            if not image_base64 or len(image_base64) < 100:
                return jsonify({"response": "❌ Image invalide ou trop petite."})

            media_type = get_image_media_type(image_base64)
            question   = user_message.strip() if user_message.strip() else "Décris cette image en détail en français."

            print(f"[IMAGE ANALYSIS] media_type={media_type} b64_len={len(image_base64)}")

            r = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                        {"type": "text", "text": question},
                    ],
                }],
                max_tokens=1024,
            )
            return jsonify({"response": r.choices[0].message.content})

        except Exception as e:
            print("[IMAGE ANALYSIS ERROR]", e)
            try:
                r = client.chat.completions.create(
                    model="meta-llama/llama-4-maverick-17b-128e-instruct",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                            {"type": "text", "text": question},
                        ],
                    }],
                    max_tokens=1024,
                )
                return jsonify({"response": r.choices[0].message.content})
            except Exception as e2:
                print("[IMAGE FALLBACK ERROR]", e2)
                return jsonify({"response": f"❌ Analyse image impossible : {str(e)}"})

    # suite inchangée...
