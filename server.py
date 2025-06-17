import os, csv, tempfile
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import assemblyai as aai

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile, api_key: str = Form(...)):
    aai.settings.api_key = api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=False,
        language_code="en"
    ))

    transcript = transcriber.transcribe(filepath)

    if not transcript.utterances:
        txt_path = filepath.replace(".mp4", "_plain.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript.text)
        return FileResponse(txt_path, filename="transcript.txt")

    csv_path = filepath.replace(".mp4", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for u in transcript.utterances:
            writer.writerow([u.speaker, round(u.start/1000, 2), round(u.end/1000, 2), u.text])

    return FileResponse(csv_path, filename="transcript.csv")

