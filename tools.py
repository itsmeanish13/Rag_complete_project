"""
tools.py — Groq tool definitions and execution against a Corpus.

Groq uses the OpenAI function-calling format:
  - Tools are passed as {"type": "function", "function": {...}}
  - Responses use finish_reason == "tool_calls" and a `tool_calls` list
  - Each call has .function.name and .function.arguments (a JSON string)

Three tools are registered:
  search_corpus  — TF-IDF retrieval over the loaded chunks
  get_chunk      — fetch one chunk by ID
  list_sources   — list ingested files and chunk counts
"""

from __future__ import annotations

import json
from typing import Any

from corpus import Corpus


# ── Tool schemas (OpenAI / Groq format) ──────────────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "strict": True,
            "description": (
                "Search the loaded document corpus and return the most relevant passages "
                "for a given query. Always call this before answering questions about "
                "the documents. Prefer retrieved evidence over answering from memory."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "description": "A concise retrieval query (keywords or a short sentence).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk",
            "strict": True,
            "description": (
                "Fetch the full text of a specific chunk by its ID. "
                "Use when a search result looks relevant but the snippet is truncated."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID from search_corpus (e.g. 'doc0_chunk3').",
                    }
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "strict": True,
            "description": "List all documents loaded into the corpus and their chunk counts.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    },
]


# ── Executor ──────────────────────────────────────────────────────────────────

class ToolExecutor:
    """Runs Groq tool calls against a live Corpus."""

    def __init__(self, corpus: Corpus) -> None:
        self.corpus = corpus

    def run(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call and return the result as a JSON string."""
        try:
            if tool_name == "search_corpus":
                return self._search_corpus(**tool_input)
            elif tool_name == "get_chunk":
                return self._get_chunk(**tool_input)
            elif tool_name == "list_sources":
                return self._list_sources()
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    # ── Tool implementations ──────────────────────────────────────────────────

    def _search_corpus(self, query: str, top_k: int = 4) -> str:
        if not query or not str(query).strip():
            return json.dumps({"results": [], "message": "No query was provided."})
        if isinstance(top_k, str):
            top_k = int(top_k)
        top_k = max(1, min(10, top_k))
        results = self.corpus.search(query, top_k=top_k)
        if not results:
            return json.dumps({"results": [], "message": "No relevant passages found."})

        formatted = [
            {
                "chunk_id": chunk.id,
                "source": chunk.source,
                "page": chunk.page,
                "relevance_score": round(score, 4),
                "text": chunk.text,
            }
            for chunk, score in results
        ]
        return json.dumps({"results": formatted})

    def _get_chunk(self, chunk_id: str) -> str:
        for chunk in self.corpus.chunks:
            if chunk.id == chunk_id:
                return json.dumps(
                    {
                        "chunk_id": chunk.id,
                        "source": chunk.source,
                        "page": chunk.page,
                        "text": chunk.text,
                    }
                )
        return json.dumps({"error": f"Chunk not found: {chunk_id}"})

    def _list_sources(self) -> str:
        return json.dumps(self.corpus.stats())