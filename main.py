"""
main.py — FastAPI backend for the RAG + Voice web app.

Endpoints:
  GET  /                   → serves index.html
  GET  /api/status         → corpus stats + ready flag
  POST /api/upload         → ingest uploaded PDF / .txt files
  POST /api/chat           → text question → RAG answer (JSON)
  POST /api/transcribe     → audio blob → transcribed text (Groq Whisper)
  POST /api/tts            → text → MP3 audio bytes (gTTS)
  POST /api/clear          → wipe conversation history
  GET  /api/history        → return conversation history

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import groq as groq_lib
from groq import Groq

from audio import transcribe, _strip_markdown
from corpus import Corpus
from tools import TOOL_DEFINITIONS, ToolExecutor

from dotenv import load_dotenv
load_dotenv()



# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Voice Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html) from same directory
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

# ── Global state (single-user; extend with sessions for multi-user) ───────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS    = 1024

_client:   Groq | None        = None
_corpus:   Corpus              = Corpus()
_executor: ToolExecutor | None = None
_history:  list[dict[str, Any]] = []

SYSTEM_PROMPT = """You are a helpful assistant answering questions from a loaded \
document corpus. Relevant passages will be provided to you in the prompt.

Rules:
- Cite the source filename and page when using retrieved content.
- Keep answers clear and well-structured.
- If nothing relevant is found, say so and suggest rephrasing."""


def _looks_like_document_question(user_message: str) -> bool:
    text = user_message.lower().strip()
    if not text:
        return False

    document_markers = (
        "document",
        "corpus",
        "file",
        "pdf",
        "page",
        "source",
        "quote",
        "summar",
        "according to",
        "in the text",
        "in the document",
        "loaded",
        "retriev",
    )
    if any(marker in text for marker in document_markers):
        return True

    word_count = len(text.split())
    if word_count <= 3:
        return False

    return text.endswith("?")


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _build_retrieval_context(query: str, corpus: Corpus, top_k: int = 4) -> tuple[str, list[dict[str, Any]]]:
    if not corpus.chunks:
        return "No corpus passages are currently available.", []

    results = corpus.search(query, top_k=top_k)
    if not results:
        return "No relevant passages were found in the corpus.", []

    snippets: list[str] = []
    tool_events: list[dict[str, Any]] = []

    for chunk, score in results:
        location = chunk.source if chunk.page is None else f"{chunk.source} p.{chunk.page}"
        snippet_text = chunk.text[:240].replace("\n", " ")
        snippets.append(f"- {location} | score={score:.4f} | {snippet_text}")
        tool_events.append(
            {
                "tool": "search_corpus",
                "query": query,
                "snippets": [f"{location} — {snippet_text}…"],
            }
        )

    context = "Retrieved corpus passages:\n" + "\n".join(snippets)
    return context, tool_events


def _get_client() -> Groq:
    global _client
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not set. Set it as an environment variable.",
        )
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _get_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor(_corpus)
    return _executor


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    index = BASE_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index))


@app.get("/api/status")
async def status():
    stats = _corpus.stats()
    return {
        "ready":      bool(GROQ_API_KEY),
        "model":      DEFAULT_MODEL,
        "corpus":     stats,
        "history_len": len(_history),
    }


# ── File upload & ingestion ───────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    global _executor
    results = []

    for upload in files:
        suffix = Path(upload.filename or "file.txt").suffix.lower()
        if suffix not in (".pdf", ".txt", ".md", ".rst"):
            results.append({"file": upload.filename, "error": "Unsupported type"})
            continue

        # Write to a temp file so corpus.add_file() can read it
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        try:
            n = _corpus.add_file(tmp_path, chunk_size=600, overlap=100)
            # Rename source to original filename in corpus chunks
            for chunk in _corpus.chunks:
                if chunk.source == Path(tmp_path).name:
                    chunk.source = upload.filename or chunk.source
            _corpus._rebuild_index()
            _executor = ToolExecutor(_corpus)   # refresh after new data
            results.append({"file": upload.filename, "chunks": n})
        except Exception as exc:
            results.append({"file": upload.filename, "error": str(exc)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return {"uploaded": results, "corpus": _corpus.stats()}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model:   str = DEFAULT_MODEL


@app.post("/api/chat")
async def chat(req: ChatRequest):
    client   = _get_client()
    corpus   = _corpus

    _history.append({"role": "user", "content": req.message})

    context_text, tool_events = _build_retrieval_context(req.message, corpus)

    try:
        response = client.chat.completions.create(
            model=req.model,
            max_completion_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": context_text},
            ] + _history,
        )

        msg = response.choices[0].message
        answer = msg.content or ""
        _history.append({
            "role": "assistant",
            "content": answer,
        })

    except groq_lib.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY")
    except groq_lib.RateLimitError:
        raise HTTPException(status_code=429, detail="Groq rate limit hit — wait a moment")
    except groq_lib.BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc.message))
    except groq_lib.APIConnectionError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot reach Groq API: {exc}")
    except groq_lib.APIStatusError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

    return {"answer": answer, "tool_events": tool_events}


# ── Transcription (Groq Whisper) ──────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    client = _get_client()
    wav_bytes = await audio.read()
    try:
        text = transcribe(wav_bytes, client, language="en")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    return {"text": text}


# ── Text-to-speech (gTTS → MP3 bytes) ────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    try:
        from gtts import gTTS
    except ImportError:
        raise HTTPException(status_code=501, detail="gTTS not installed: pip install gtts")

    clean = _strip_markdown(req.text)
    tts   = gTTS(text=clean, lang=req.lang)
    buf   = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/mpeg")


# ── History ───────────────────────────────────────────────────────────────────

@app.get("/api/history")
async def get_history():
    visible = [
        {"role": t["role"], "content": t.get("content", "")}
        for t in _history
        if t["role"] in ("user", "assistant") and t.get("content")
    ]
    return {"history": visible}


@app.post("/api/clear")
async def clear_history():
    _history.clear()
    return {"cleared": True}