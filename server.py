from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import assemblyai as aai
import tempfile
import os
import csv

# Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS so Lovable can call this from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: replace with your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Transcription endpoint
@app.post("/transcribe")
async def transcribe(file: UploadFile, api_key: str = Form(...)):
    aai.settings.api_key = api_key

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    # Configure transcription
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=False,
        language_code="en"
    ))

    # Transcribe
    transcript = transcriber.transcribe(filepath)

    # Fallback to plain text if no speaker labels
    if not transcript.utterances:
        txt_path = filepath.replace(".mp4", "_plain.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.text)
        return FileResponse(txt_path, filename="transcript.txt")

    # Save .csv with speaker labels and timestamps
    csv_path = filepath.replace(".mp4", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for u in transcript.utterances:
            writer.writerow([
                u.speaker,
                round(u.start / 1000, 2),
                round(u.end / 1000, 2),
                u.text
            ])

    return FileResponse(csv_path, filename="transcript.csv")

