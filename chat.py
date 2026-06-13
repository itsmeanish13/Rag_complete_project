"""
chat.py — Terminal RAG chatbot powered by Groq with multi-turn history and tool calling.

Groq uses the OpenAI function-calling protocol:
  • finish_reason == "tool_calls"  → execute tools, feed results back
  • finish_reason == "stop"        → final answer, print and loop

Usage:
    python chat.py sample_corpus.txt
    python chat.py docs/report.pdf notes.txt
    python chat.py *.pdf --model gemma2-9b-it --chunk-size 400 --top-k 6
    python chat.py --help
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

from corpus import Corpus
from tools import TOOL_DEFINITIONS, ToolExecutor
from dotenv import load_dotenv
load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "llama-3.3-70b-versatile"
MAX_TOKENS     = 1024
WRAP_WIDTH     = 88

# Models that support tool calling on Groq's free tier
TOOL_CAPABLE_MODELS = {
    "llama-3.3-70b-versatile":          "Best overall quality — recommended",
    "llama-3.1-8b-instant":             "Fastest, lowest quota usage",
    "llama-4-scout-17b-16e-instruct":   "512K context — great for long docs",
    "gemma2-9b-it":                     "Fast and efficient",
}

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions strictly \
from a loaded document corpus. You have three tools available:

• search_corpus  — find relevant passages by keyword/semantic query (use the default retrieval depth)
• get_chunk      — fetch the full text of a specific chunk by ID
• list_sources   — see which documents are loaded

Rules:
- Always search the corpus before answering document questions.
- When calling search_corpus, pass only the query; the tool uses its default number of results.
- Cite your sources: mention the filename and page number when quoting a passage.
- For multi-part questions, run one search per sub-question.
- If nothing relevant is found, say so and explain what you searched for.
- If the question is clearly about general knowledge, answer from your own knowledge \
and say so explicitly."""


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

CYAN  = "\033[96m"
GREEN = "\033[92m"
DIM   = "\033[2m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _wrap(text: str, indent: str = "") -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.append(textwrap.fill(paragraph, width=WRAP_WIDTH, subsequent_indent=indent))
    return "\n".join(lines)


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET}")
    print(_wrap(text, indent="  "))
    print()


def print_tool_call(name: str, inputs: dict) -> None:
    query = inputs.get("query", inputs.get("chunk_id", ""))
    print(f"  {DIM}⚙  {name}({query!r}){RESET}")


def print_tool_snippet(result_json: str) -> None:
    try:
        data = json.loads(result_json)
        results = data.get("results", [])
        if results:
            for r in results:
                src = r["source"] + (f" p.{r['page']}" if r.get("page") else "")
                snippet = r["text"][:120].replace("\n", " ")
                print(f"  {DIM}  ↳ [{r['chunk_id']}] {src} — {snippet}…{RESET}")
        elif "error" in data:
            print(f"  {DIM}  ↳ Error: {data['error']}{RESET}")
        else:
            print(f"  {DIM}  ↳ {result_json[:160]}{RESET}")
    except Exception:
        print(f"  {DIM}  ↳ {result_json[:160]}{RESET}")


def print_banner(model: str, stats: dict) -> None:
    sources = stats.get("sources", {})
    total   = stats.get("total_chunks", 0)
    print(f"\n{CYAN}{BOLD}╔═══════════════════════════════════════╗")
    print(f"║    Groq RAG Terminal Chatbot  ⚡      ║")
    print(f"╚═══════════════════════════════════════╝{RESET}")
    print(f"\n{BOLD}Model:{RESET}  {model}")
    print(f"{BOLD}Corpus:{RESET}")
    for name, count in sources.items():
        print(f"  • {name}  ({count} chunks)")
    print(f"  Total: {total} chunks indexed\n")
    print(f'{DIM}Type your question, or "exit" / Ctrl-C to quit.{RESET}')
    print(f'{DIM}Commands: history | clear | exit{RESET}\n')


# ── Agentic turn (Groq / OpenAI tool-call loop) ───────────────────────────────

def run_turn(
    client: Groq,
    executor: ToolExecutor,
    model: str,
    history: list[dict[str, Any]],
    user_message: str,
) -> str:
    """
    Run one user turn, looping through tool calls until Groq returns a
    stop completion. History is mutated in place.

    Groq tool-call protocol:
      response.choices[0].finish_reason == "tool_calls"
        → response.choices[0].message.tool_calls  (list of ChatCompletionMessageToolCall)
            .id            — tool call ID (echo back in tool result)
            .function.name — tool name
            .function.arguments — JSON string of inputs

      Tool results are appended as role="tool" messages:
        {"role": "tool", "tool_call_id": <id>, "content": <result_str>}
    """
    history.append({"role": "user", "content": user_message})

    while True:
        use_tools = _looks_like_document_question(user_message)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "max_completion_tokens": MAX_TOKENS,
            "tool_choice": "auto" if use_tools else "none",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        }
        if use_tools:
            request_kwargs["tools"] = TOOL_DEFINITIONS

        response = client.chat.completions.create(
            **request_kwargs,
        )

        choice = response.choices[0]
        msg    = choice.message

        # ── Append assistant turn to history ──────────────────────────────────
        # Build a plain dict so history stays serialisable across turns.
        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
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

        # ── No more tool calls → return final answer ──────────────────────────
        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        # ── Execute every tool call in this response ──────────────────────────
        for tc in msg.tool_calls:
            name   = tc.function.name
            inputs = json.loads(tc.function.arguments or "{}")

            print_tool_call(name, inputs)
            result = executor.run(name, inputs)
            print_tool_snippet(result)

            history.append(
                {
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                }
            )
        # Loop back — Groq will now synthesise using the tool results


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Terminal RAG chatbot powered by Groq tool calling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("files", nargs="+", metavar="FILE",
                   help="PDF or .txt files to load into the corpus")
    p.add_argument("--model",      "-m", default=DEFAULT_MODEL,
                   help=f"Groq model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--chunk-size", type=int, default=600,
                   help="Words per chunk (default: 600)")
    p.add_argument("--overlap",    type=int, default=100,
                   help="Word overlap between adjacent chunks (default: 100)")
    p.add_argument("--top-k",      type=int, default=4,
                   help="Chunks returned per search (default: 4)")
    return p


def _print_history(history: list[dict]) -> None:
    print(f"\n{DIM}── Conversation history ({len(history)} turns) ──")
    for turn in history:
        role = turn["role"].upper()
        content = turn.get("content", "")
        tool_calls = turn.get("tool_calls", [])
        if content:
            print(f"  {role}: {str(content)[:90]}")
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"  {role}: [tool_call] {fn.get('name', '?')}({fn.get('arguments', '')[:60]})")
    print(f"──{RESET}\n")


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    # ── API key ───────────────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: GROQ_API_KEY is not set.\n"
            "  Get a free key at: https://console.groq.com/keys\n"
            "  Then: export GROQ_API_KEY='gsk_...'",
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
        print("Error: no files were loaded. Exiting.", file=sys.stderr)
        return 1

    executor = ToolExecutor(corpus)
    client   = Groq(api_key=api_key)
    history: list[dict[str, Any]] = []

    print_banner(args.model, corpus.stats())

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye", ":q"}:
            print("Bye!")
            return 0
        if user_input.lower() in {"history", ":history"}:
            _print_history(history)
            continue
        if user_input.lower() in {"clear", ":clear"}:
            history.clear()
            print(f"  {DIM}Conversation history cleared.{RESET}\n")
            continue

        try:
            answer = run_turn(client, executor, args.model, history, user_input)
            print_assistant(answer)

        except groq_lib.AuthenticationError:
            print("Authentication failed — check your GROQ_API_KEY.", file=sys.stderr)
            return 1
        except groq_lib.RateLimitError:
            print(
                "Rate limit hit (30 req/min, 6K tok/min on free tier).\n"
                "Wait a moment, or switch to --model llama-3.1-8b-instant.",
                file=sys.stderr,
            )
        except groq_lib.BadRequestError as exc:
            print(f"Bad request: {exc.message}", file=sys.stderr)
        except groq_lib.APIConnectionError as exc:
            print(f"Connection error: {exc}", file=sys.stderr)
        except groq_lib.APIStatusError as exc:
            print(f"API error {exc.status_code}: {exc.message}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())