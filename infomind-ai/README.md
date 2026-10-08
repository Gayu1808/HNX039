# InfoMind AI — Intelligent Information Understanding & Decision Assistant
HACKNEXT'26 · PS01 Smart Automation · FIND → UNDERSTAND → ORGANIZE → USE

## Run
```
pip install -r requirements.txt
uvicorn app:app --app-dir backend --reload      # open http://localhost:8000
```
Login `admin` / `admin123` (set `ADMIN_PASSWORD` env var before first run to change). Click **Explore demo** to load 5 sample documents.
Tests: `pytest tests`. Scanned images need the Tesseract OCR binary installed.

## AI configuration
- Default: fully local — TF-IDF semantic retrieval (scikit-learn) + rule/regex extraction. No key needed.
- Optional LLM answers: set `ANTHROPIC_API_KEY` (and `LLM_MODEL`) in `.env.example` → env. Answers stay restricted to retrieved chunks; anything unsupported returns "I couldn't find this information in the uploaded documents."
- To upgrade retrieval, swap `retrieve()` in `backend/app.py` for sentence-transformers + FAISS/Chroma.

## Architecture
Upload → extract (PyPDF/python-docx/OCR) → chunk by paragraph/page → index → field/entity extraction → findings (deadline, missing, conflict) → search / Q&A / compare → Action Center.
Tables: users, docs, chunks, findings, actions, searches (SQLite). Endpoints: see `/docs` (FastAPI auto API docs).
Honest limits: missing-item checks only run on application-type docs against a fixed checklist; conflicts compare `Key: value` lines across documents.
