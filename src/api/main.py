

from fastapi import FastAPI

app = FastAPI(title="Local RAG Chatbot", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    # Simple health endpoint for verifying the service is running.
    return {"status": "ok"}
