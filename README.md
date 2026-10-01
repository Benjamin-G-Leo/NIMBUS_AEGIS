# NIMBUS_AEGIS
AEGIS is a dynamic inventory managing tool

## ReliefMatch orchestration engine

FastAPI + OpenRouter (Qwen) layer that triages free-text offers, computes the
allocation grid via `core.py`, and serves the hand-coded frontend in `static/`.

### Files
- `app.py`     — FastAPI skeleton (serves the UI, the `/api/*` endpoints, and mounts `static/` at `/static`).
- `server.py`  — FastMCP inspector exposing `evaluate_offer`, `detect_clashes`, `apply_reallocation`.
- `core.py`    — deterministic state math (gaps, clashes, reallocation).
- `static/`    — the hand-coded frontend (mounted at `/static`; `/` serves `index.html`).

### Setup
```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Configure OpenRouter (Qwen)
```bash
export OPENROUTER_API_KEY="sk-or-..."
# optional overrides:
export OPENROUTER_MODEL="qwen/qwen-3-30b-a3b"
```
If the key is missing or the call fails, triage automatically degrades to an
offline regex parser and a template fallback email (no hard failure).

### Run the API + UI
```bash
./.venv/bin/python -m uvicorn app:app --port 8000
# open http://localhost:8000/
```
Endpoints: `GET /api/grid_state`, `POST /api/triage`, `POST /api/reallocate`.

### Run the MCP inspector
```bash
./.venv/bin/python server.py                      # stdio (default)
MCP_TRANSPORT=streamable-http MCP_PORT=8001 ./.venv/bin/python server.py
```

