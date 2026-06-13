"""
voice_agent.py — Fully voice-to-voice RAG agent.

Pipeline per turn:
  1. record_question()   — capture mic audio until silence
  2. transcribe()        — Groq Whisper → text
  3. run_turn()          — Groq LLM + TF-IDF corpus retrieval via tool calling
  4. speak()             — gTTS → MP3 → system audio player

Usage:
    python voice_agent.py sample_corpus.txt
    python voice_agent.py docs/report.pdf notes.txt
    python voice_agent.py *.pdf --model llama-3.1-8b-instant --lang en
    python voice_agent.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any

import groq as groq_lib
from groq import Groq

from audio import record_question, speak, transcribe
from corpus import Corpus
from tools import TOOL_DEFINITIONS, ToolExecutor

from dotenv import load_dotenv
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS    = 1024
WRAP_WIDTH    = 88

SYSTEM_PROMPT = """You are a voice assistant that answers questions from a loaded \
document corpus. You have three tools:

• search_corpus  — find relevant passages by keyword/semantic query
• get_chunk      — fetch the full text of a specific chunk by ID
• list_sources   — see which documents are loaded

Rules:
- Always search the corpus before answering document questions.
- When calling search_corpus, provide a non-empty JSON object with a query string.
- Never call search_corpus without a query.
- Keep answers concise and natural for spoken delivery (2-4 sentences unless \
the user explicitly asks for detail).
- Do NOT use markdown, bullet points, or symbols in your answer — it will be \
read aloud.
- Cite the source document by name (not chunk IDs) when using retrieved content.
- If nothing relevant is found, say so briefly and suggest rephrasing.
- For general knowledge questions, answer from your own knowledge and say so."""


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


# ── Colour helpers ────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _wrap(text: str) -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.append(textwrap.fill(para, width=WRAP_WIDTH, subsequent_indent="  "))
        else:
            lines.append("")
    return "\n".join(lines)


def print_banner(model: str, stats: dict) -> None:
    sources = stats.get("sources", {})
    total   = stats.get("total_chunks", 0)
    print(f"\n{CYAN}{BOLD}╔════════════════════════════════════════╗")
    print(f"║   Groq Voice RAG Agent  🎙 ⚡ 🔊       ║")
    print(f"╚════════════════════════════════════════╝{RESET}")
    print(f"\n{BOLD}STT:{RESET}    Groq Whisper (whisper-large-v3-turbo)")
    print(f"{BOLD}LLM:{RESET}    {model}")
    print(f"{BOLD}TTS:{RESET}    gTTS → system audio player")
    print(f"\n{BOLD}Corpus:{RESET}")
    for name, count in sources.items():
        print(f"  • {name}  ({count} chunks)")
    print(f"  Total: {total} chunks indexed\n")
    print(f"{DIM}Press Enter to start recording. Say 'exit' or 'quit' to stop.")
    print(f"Text commands: history | clear | exit{RESET}\n")


def print_user(text: str) -> None:
    print(f"\n{CYAN}{BOLD}You (heard):{RESET}  {text}")


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET}")
    print(_wrap(text))
    print()


def print_tool_call(name: str, inputs: dict) -> None:
    query = inputs.get("query", inputs.get("chunk_id", ""))
    print(f"  {DIM}⚙  {name}({query!r}){RESET}")


def print_tool_snippet(result_json: str) -> None:
    try:
        data    = json.loads(result_json)
        results = data.get("results", [])
        if results:
            for r in results:
                src     = r["source"] + (f" p.{r['page']}" if r.get("page") else "")
                snippet = r["text"][:100].replace("\n", " ")
                print(f"  {DIM}  ↳ [{r['chunk_id']}] {src} — {snippet}…{RESET}")
        elif "error" in data:
            print(f"  {DIM}  ↳ Error: {data['error']}{RESET}")
        else:
            print(f"  {DIM}  ↳ {result_json[:160]}{RESET}")
    except Exception:
        print(f"  {DIM}  ↳ {result_json[:160]}{RESET}")


# ── Agentic RAG turn (identical protocol to groq_rag_chatbot) ─────────────────

def run_turn(
    client: Groq,
    executor: ToolExecutor,
    model: str,
    history: list[dict[str, Any]],
    user_message: str,
) -> str:
    history.append({"role": "user", "content": user_message})
    use_tools = _looks_like_document_question(user_message)

    while True:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": MAX_TOKENS,
            "tool_choice": "auto" if use_tools else "none",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        }
        if use_tools:
            request_kwargs["tools"] = TOOL_DEFINITIONS

        response = client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]
        msg    = choice.message

        assistant_entry: dict[str, Any] = {
            "role":    "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        history.append(assistant_entry)

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            name   = tc.function.name
            inputs = json.loads(tc.function.arguments or "{}")
            print_tool_call(name, inputs)
            result = executor.run(name, inputs)
            print_tool_snippet(result)
            history.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })


# ── History display ───────────────────────────────────────────────────────────

def _print_history(history: list[dict]) -> None:
    print(f"\n{DIM}── Conversation history ({len(history)} turns) ──")
    for turn in history:
        role       = turn["role"].upper()
        content    = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])
        if content:
            print(f"  {role}: {str(content)[:90]}")
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"  {role}: [tool_call] {fn.get('name')}({fn.get('arguments','')[:50]})")
    print(f"──{RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Voice-to-voice RAG agent: speak → Groq Whisper → LLM → gTTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="PDF or .txt files to load into the corpus")
    p.add_argument("--model",      "-m", default=DEFAULT_MODEL,
                   help=f"Groq LLM model (default: {DEFAULT_MODEL})")
    p.add_argument("--lang",       "-l", default="en",
                   help="Language code for STT + TTS (default: en)")
    p.add_argument("--chunk-size", type=int, default=600,
                   help="Words per corpus chunk (default: 600)")
    p.add_argument("--overlap",    type=int, default=100,
                   help="Overlap between chunks in words (default: 100)")
    p.add_argument("--no-voice-out", action="store_true",
                   help="Disable TTS — print the answer only (text mode)")
    p.add_argument("--no-voice-in",  action="store_true",
                   help="Disable STT — type questions instead of speaking")
    return p


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: GROQ_API_KEY is not set.\n"
            "  Get a free key: https://console.groq.com/keys\n"
            "  Export:  export GROQ_API_KEY='gsk_...'",
            file=sys.stderr,
        )
        return 1

    # ── Build corpus ──────────────────────────────────────────────────────────
    corpus = Corpus()
    print()
    for filepath in args.files:
        try:
            n = corpus.add_file(filepath, chunk_size=args.chunk_size, overlap=args.overlap)
            print(f"  Loaded {filepath!r}  →  {n} chunks")
        except FileNotFoundError:
            print(f"  Warning: file not found, skipping: {filepath}", file=sys.stderr)
        except Exception as exc:
            print(f"  Warning: could not load {filepath!r}: {exc}", file=sys.stderr)

    if not corpus.chunks:
        print("Error: no files loaded. Exiting.", file=sys.stderr)
        return 1

    client   = Groq(api_key=api_key)
    executor = ToolExecutor(corpus)
    history:  list[dict[str, Any]] = []

    print_banner(args.model, corpus.stats())

    # ── Voice REPL ────────────────────────────────────────────────────────────
    while True:
        # ── Input: microphone or keyboard ─────────────────────────────────────
        if args.no_voice_in:
            try:
                raw = input(f"{CYAN}{BOLD}You:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return 0
            user_text = raw
        else:
            try:
                input(f"{YELLOW}[Enter]{RESET} to speak  "
                      f"{DIM}(or type 'exit'){RESET}  → ")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return 0

            # Allow a typed shortcut at the prompt
            # (input() returns the typed text on Enter)
            # We re-check stdin for typed commands via a second path:
            # Actually: the input() above returns '' when user just hits Enter.
            # A typed word at that prompt returns the typed word.
            # We'll handle it cleanly below.

            # Record audio
            try:
                wav_bytes = record_question()
            except ImportError as exc:
                print(f"\nMicrophone error: {exc}", file=sys.stderr)
                return 1
            except OSError as exc:
                print(f"\nAudio device error: {exc}", file=sys.stderr)
                return 1

            # Transcribe
            print(f"  {DIM}Transcribing…{RESET}", end="", flush=True)
            try:
                user_text = transcribe(wav_bytes, client, language=args.lang)
            except Exception as exc:
                print(f"\n  Transcription failed: {exc}", file=sys.stderr)
                continue

            print(f"\r  {DIM}               {RESET}", end="\r")  # clear line
            print_user(user_text)

        # ── Typed command shortcuts ───────────────────────────────────────────
        cmd = user_text.lower().strip()
        if not cmd:
            continue
        if cmd in {"exit", "quit", "bye"}:
            print("Bye!")
            if not args.no_voice_out:
                try:
                    speak("Goodbye!", lang=args.lang)
                except Exception:
                    pass
            return 0
        if cmd in {"history", ":history"}:
            _print_history(history)
            continue
        if cmd in {"clear", ":clear"}:
            history.clear()
            print(f"  {DIM}History cleared.{RESET}\n")
            continue

        # ── RAG turn ──────────────────────────────────────────────────────────
        try:
            print(f"  {DIM}Thinking…{RESET}", flush=True)
            answer = run_turn(client, executor, args.model, history, user_text)
        except groq_lib.AuthenticationError:
            print("Authentication failed — check GROQ_API_KEY.", file=sys.stderr)
            return 1
        except groq_lib.RateLimitError:
            msg = "Rate limit hit. Please wait a moment."
            print(f"\n  {msg}", file=sys.stderr)
            if not args.no_voice_out:
                try:
                    speak(msg, lang=args.lang)
                except Exception:
                    pass
            continue
        except groq_lib.BadRequestError as exc:
            print(f"\n  Bad request: {exc.message}", file=sys.stderr)
            continue
        except groq_lib.APIConnectionError as exc:
            print(f"\n  Connection error: {exc}", file=sys.stderr)
            continue
        except groq_lib.APIStatusError as exc:
            print(f"\n  API error {exc.status_code}: {exc.message}", file=sys.stderr)
            continue

        # ── Output: print + speak ─────────────────────────────────────────────
        print_assistant(answer)

        if not args.no_voice_out:
            print(f"  {DIM}Speaking…{RESET}", flush=True)
            try:
                speak(answer, lang=args.lang)
            except ImportError as exc:
                print(f"  TTS error: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"  TTS playback error: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())