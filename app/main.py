from fastapi import FastAPI

app = FastAPI(title="Helpdesk Search Assistant")

@app.get("/health")
def health():
    return {"status": "ok"}
