from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pathlib import Path
import uuid

from app.search import load_pdf, search

app = FastAPI(title="Helpdesk Search Assistant")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def index():
    return("Welcome to the API -Balka :)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
async def search_pdf(
    file: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=2000),
    top_k: int = Form(3, ge=1, le=20),
):
    # Strip removes leading and trailing whitespace, including spaces and newlines.
    question = question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    # Save uploaded PDF
    file_id = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"

    pdf_path.write_bytes(contents)

    # Run semantic search
    try:
        pages = load_pdf(str(pdf_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="The PDF has no extractable text. Scanned PDFs may need OCR.",
        )

    results = search(pages, question, top_k=top_k)

    return {
        "question": question,
        "top_k": top_k,
        "results": results,
    }
