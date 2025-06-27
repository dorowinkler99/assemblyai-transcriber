
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import assemblyai as aai
import tempfile
import os
import csv

app = FastAPI()

# Allow all frontend clients (e.g. Lovable) for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    api_key: str = Form(...),
    language_code: str = Form("en")
):
    aai.settings.api_key = api_key

    # Save uploaded file to disk
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    # Transcribe synchronously with disfluencies enabled ✅
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=True,  # ✅ this enables "umm", "uh", etc.
        language_code=language_code
    ))

    transcript = transcriber.transcribe(filepath)

    # Fallback: plain text if no utterances
    if not transcript.utterances:
        txt_path = filepath.replace(suffix, "_plain.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.text)
        return FileResponse(txt_path, filename="transcript.txt")

    # Export to CSV
    csv_path = filepath.replace(suffix, ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for utt in sorted(transcript.utterances, key=lambda x: x.start):
            writer.writerow([
                utt.speaker,
                round(utt.start / 1000, 2),
                round(utt.end / 1000, 2),
                utt.text
            ])

    return FileResponse(csv_path, filename="transcript.csv")

