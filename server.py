from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import assemblyai as aai
import tempfile
import os

app = FastAPI()

# ✅ Replace with your actual Lovable frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow any frontend for now
    allow_credentials=False,  # must be false when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile,
    api_key: str = Form(...),
    language_code: str = Form("en"),
    export_format: str = Form("xlsx")  # Default to Excel
):
    # Set AssemblyAI API key
    aai.settings.api_key = api_key

    # Save uploaded file to temp location
    file_ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    # Configure transcription
    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=False,
        language_code=language_code
    ))

    # Transcribe
    transcript = transcriber.transcribe(filepath)

    # Fallback to plain text if no utterances
    if not transcript.utterances:
        txt_path = filepath.replace(file_ext, "_plain.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.text)
        return FileResponse(txt_path, filename="transcript.txt")

    # === CSV Export ===
    if export_format == "csv":
        import csv
        csv_path = filepath.replace(file_ext, ".csv")
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

    # === Excel Export ===
    else:
        from openpyxl import Workbook
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
        xlsx_path = filepath.replace(file_ext, ".xlsx")
        wb.save(xlsx_path)
        return FileResponse(xlsx_path, filename="transcript.xlsx")

