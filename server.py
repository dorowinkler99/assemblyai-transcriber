from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import assemblyai as aai
import tempfile
import os
import csv

app = FastAPI()

# CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job cache to keep track of file paths per job
job_files = {}

@app.post("/transcribe")
async def start_transcription(
    file: UploadFile,
    api_key: str = Form(...),
    language_code: str = Form("en")
):
    aai.settings.api_key = api_key
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        filepath = tmp.name

    transcriber = aai.Transcriber(config=aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True,
        punctuate=True,
        format_text=True,
        disfluencies=False,
        language_code=language_code
    ))

    job = transcriber.submit(filepath)
    job_files[job.id] = filepath
    return {"job_id": job.id, "status": "submitted"}

@app.get("/status/{job_id}")
async def check_status(job_id: str):
    poller = aai.Transcriber()
    transcript = poller.poll(job_id)

    if transcript.status == "completed":
        return {"status": "completed"}
    elif transcript.status == "failed":
        return {"status": "failed", "error": transcript.error}
    else:
        return {"status": transcript.status}

@app.get("/download/{job_id}")
async def download_csv(job_id: str):
    poller = aai.Transcriber()
    transcript = poller.poll(job_id)

    if transcript.status != "completed":
        return JSONResponse(status_code=202, content={"message": "Transcript not ready yet."})

    if job_id not in job_files:
        return JSONResponse(status_code=404, content={"message": "Original file not found."})

    base_path = job_files[job_id]
    csv_path = base_path.replace(os.path.splitext(base_path)[1], ".csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Speaker", "Start", "End", "Text"])
        for u in sorted(transcript.utterances, key=lambda x: x.start):
            writer.writerow([
                u.speaker,
                round(u.start / 1000, 2),
                round(u.end / 1000, 2),
                u.text
            ])

    return FileResponse(csv_path, filename="transcript.csv")

