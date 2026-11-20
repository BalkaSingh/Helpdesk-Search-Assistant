# Helpdesk-Search-Assistant

This project is a FastAPI backend for semantic search over PDF documents. A user can upload a PDF through the `/search` endpoint, ask a question about the file, and get back the most relevant passages based on meaning instead of simple keyword matching.

The API returns:
- the question that was asked
- the `top_k` value used for retrieval
- the most relevant results from the uploaded PDF

# Environment Setup

1. Install Python 3.11 if needed, then open a new PowerShell terminal. From the
repo root, create a virtual environment (only needed for initial setup):

```powershell
python -m venv .venv
```

A virtual environment keeps this project's Python packages separate from other
projects.

2. Activate the virtual environment from the repo root:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```powershell
(.venv)
```

3. Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

# Run the App

Start the FastAPI server with:

```powershell
python -m uvicorn app.main:app --reload
```

You can also run the app without activating the environment. This is useful if
PowerShell blocks `Activate.ps1`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The explicit path selects the project's Python interpreter directly.

You should then see Uvicorn running and a message:

```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

# API Endpoints

## Health Check

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

This is the easiest way to test the API in the browser.

## POST `/search`

This endpoint now allows the user to upload a PDF directly and search it in the same request.

Required form fields:
- `file`: the PDF document to upload
- `question`: the question you want to ask about the document

Optional form field:
- `top_k`: number of matching results to return, from `1` to `20`; default is `3`

Only PDF files are supported. Questions must contain non-whitespace text and
must be no longer than 2,000 characters before trimming.

Invalid requests return an HTTP error with a `detail` field:

- `400`: empty, unreadable, or password-protected PDF (or a file that is not a PDF)
- `415`: upload content type is not `application/pdf`
- `422`: invalid question or `top_k`, or a PDF with no extractable text

Scanned PDFs may contain images rather than text. Reading their words requires
OCR (optical character recognition), which this project does not yet implement.

Example using `curl`:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/search" `
  -F "file=@sample.pdf;type=application/pdf" `
  -F "question=What is this document about?" `
  -F "top_k=3"
```

Example response shape:

```json
{
  "question": "What is this document about?",
  "top_k": 3,
  "results": [
    {
      "snippet": "...",
      "score": 0.0,
      "page": 1
    }
  ]
}
```

# Test the Semantic Search Logic

There is also a local script for testing the search logic outside the API.

Place a `sample.pdf` in the repo root, then run:

```powershell
python test_search.py
```

To check validation in `/docs`, try a normal PDF with `top_k=3`, then repeat
with `top_k=-1`, a question containing only spaces, and an empty file. Expect
`200`, `422`, `422`, and `400`, respectively. A PDF with no extractable text
should return `422` instead of an unexplained empty result list.


### Swagger UI
http://127.0.0.1:8000/docs
