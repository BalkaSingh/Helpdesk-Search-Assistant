# Helpdesk-Search-Assistant
A backend application that allows users to upload PDF documents and ask questions about their content. The system searches the documents by meaning (not keywords) and returns the most relevant passages along with the document name and page number, enabling accurate and citation-based answers from internal documents.

# Environment Setup
1. From the repo root:
```
source .venv/Scripts/activate
```
### You must see:

`(.venv)`


2. Make sure dependencies are installed
```
pip install -r requirements.txt
```

# Test the semantic search logic

Place a sample.pdf in the repo root (same level as requirements.txt)

Run:
```
python test_search.py
```

Run the FastAPI server:
```
uvicorn app.main:app --reload
```

### You should see: Uvicorn running on http://127.0.0.1:8000

### Test the API

Health check: http://127.0.0.1:8000/health

Expected:

```
{"status":"ok"}
```

### Swagger UI
http://127.0.0.1:8000/docs