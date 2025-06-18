from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import assemblyai as aai
import tempfile, os
from openpyxl import Workbook

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile, api_key: str = Form(...)):
    aai.settings.api_key = api_key

    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    # Transcribe
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=False,
        language_code="en"
    ))

    transcript = transcriber.transcribe(filepath)

    # Fallback: plain text
    if not transcript.utterances:
        txt_path = filepath.replace(".mp4", "_plain.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.text)
        return FileResponse(txt_path, filename="transcript.txt")

    # ✅ Excel export with speaker diarization
    wb = Workbook()
    ws = wb.active
    ws.title = "Transcript"
    ws.append(["Speaker", "Start", "End", "Text"])

    for u in transcript.utterances:
        ws.append([
            u.speaker,
            round(u.start / 1000, 2),
            round(u.end / 1000, 2),
            u.text
        ])

    excel_path = filepath.replace(".mp4", ".xlsx")
    wb.save(excel_path)

    return FileResponse(excel_path, filename="transcript.xlsx")

