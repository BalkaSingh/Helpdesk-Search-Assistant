from fastapi import FastAPI, UploadFile, File, Form
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
    question: str = Form(...),
    top_k: int = Form(3),
):
    # Basic validation
    if file.content_type not in ("application/pdf",):
        return {"error": "Only PDF files are supported."}

    # Save uploaded PDF
    file_id = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{file_id}.pdf"

    contents = await file.read()
    pdf_path.write_bytes(contents)

    # Run semantic search
    pages = load_pdf(str(pdf_path))
    results = search(pages, question, top_k=top_k)

    return {
        "question": question,
        "top_k": top_k,
        "results": results,
    }