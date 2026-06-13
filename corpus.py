"""
corpus.py — Load, chunk, and index PDF / plain-text files for retrieval.

No external vector DB required. Uses a lightweight TF-IDF + BM25-style
scorer so the project stays dependency-minimal while still giving
meaningful retrieval results.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str           # e.g. "doc0_chunk3"
    source: str       # original filename
    page: Optional[int]
    text: str


@dataclass
class Corpus:
    chunks: list[Chunk] = field(default_factory=list)

    # Inverted index: token → {chunk_id: term_freq}
    _index: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    # IDF scores: token → float
    _idf: dict[str, float] = field(default_factory=dict)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_file(self, path: str, chunk_size: int = 600, overlap: int = 100) -> int:
        """Ingest a PDF or .txt file. Returns number of chunks added."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        ext = p.suffix.lower()
        if ext == ".pdf":
            pages = _extract_pdf(p)
        elif ext in (".txt", ".md", ".rst"):
            pages = [(None, p.read_text(encoding="utf-8", errors="replace"))]
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        doc_idx = len({c.source for c in self.chunks})
        added = 0
        for page_num, page_text in pages:
            for chunk_text in _sliding_window(page_text, chunk_size, overlap):
                cid = f"doc{doc_idx}_chunk{len(self.chunks)}"
                self.chunks.append(Chunk(id=cid, source=p.name, page=page_num, text=chunk_text))
                added += 1

        self._rebuild_index()
        return added

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the top_k chunks most relevant to *query* with their scores."""
        if not self.chunks:
            return []

        q_tokens = _tokenize(query)
        scores: dict[str, float] = defaultdict(float)

        for token in q_tokens:
            if token not in self._index:
                continue
            idf = self._idf.get(token, 0.0)
            for cid, tf in self._index[token].items():
                scores[cid] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        id_to_chunk = {c.id: c for c in self.chunks}
        return [(id_to_chunk[cid], score) for cid, score in ranked[:top_k] if cid in id_to_chunk]

    def stats(self) -> dict:
        sources = Counter(c.source for c in self.chunks)
        return {"total_chunks": len(self.chunks), "sources": dict(sources)}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        index: dict[str, dict[str, float]] = defaultdict(dict)

        for chunk in self.chunks:
            tokens = _tokenize(chunk.text)
            total = len(tokens) or 1
            tf_raw = Counter(tokens)
            for token, count in tf_raw.items():
                # Sublinear TF
                index[token][chunk.id] = (1 + math.log(count)) / total

        N = len(self.chunks)
        idf: dict[str, float] = {}
        for token, postings in index.items():
            df = len(postings)
            idf[token] = math.log((N + 1) / (df + 1)) + 1.0  # smoothed IDF

        self._index = index
        self._idf = idf


# ── Text utilities ────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "this", "that", "was",
    "are", "be", "been", "as", "i", "we", "you", "he", "she", "they",
    "have", "has", "had", "do", "does", "did", "not", "so", "if",
}


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping word-level windows."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if window:
            chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page in the PDF."""
    try:
        import pdfplumber  # preferred: layout-aware
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
        return pages
    except ImportError:
        pass

    try:
        from pypdf import PdfReader  # fallback
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
        return pages
    except ImportError:
        pass

    raise ImportError(
        "No PDF library found. Install one:\n"
        "  pip install pdfplumber   (recommended)\n"
        "  pip install pypdf        (fallback)"
    )