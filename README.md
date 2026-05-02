# Helpdesk-Search-Assistant

This project is a FastAPI backend for semantic search over PDF documents. A user can upload a PDF through the `/search` endpoint, ask a question about the file, and get back the most relevant passages based on meaning instead of simple keyword matching.

The API returns:
- the question that was asked
- the `top_k` value used for retrieval
- the most relevant results from the uploaded PDF

# Environment Setup

1. Activate the virtual environment from the repo root:

```powershell
source .venv/Scripts/activate
```

You should see:

```powershell
(.venv)
```

2. Install the dependencies:

```powershell
pip install -r requirements.txt
```

# Run the App

Start the FastAPI server with:

```powershell
uvicorn app.main:app --reload
```

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
- `top_k`: number of matching results to return, default is `3`

Only PDF files are supported.

Example using `curl`:

```powershell
curl -X POST "http://127.0.0.1:8000/search" ^
  -F "file=@sample.pdf;type=application/pdf" ^
  -F "question=What is this document about?" ^
  -F "top_k=3"
```

Example response shape:

```json
{
  "question": "What is this document about?",
  "top_k": 3,
  "results": [
    {
      "text": "...",
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


### Swagger UI
http://127.0.0.1:8000/docs