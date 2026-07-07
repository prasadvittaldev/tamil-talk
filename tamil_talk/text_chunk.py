"""Text-splitting helpers shared by the TTS synth backends: split a reply
into sentences, then sub-split any sentence longer than max_chars on clause
boundaries (commas etc.), so a single synth call never rambles past the
model's token cap.
"""
import re

_SENT_SPLIT = re.compile(r"(?<=[.?!;।])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–…])\s+")
MAX_CHUNK_CHARS = 120  # keep each synth chunk well under the ~1200-token (~14s) cap


def split_sentences(text: str) -> list:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    if parts:
        return parts
    return [text.strip()] if text.strip() else []


def _split_words(s: str, max_chars: int) -> list:
    """Last-resort hard split of an over-long clause on whitespace."""
    chunks, cur = [], ""
    for w in s.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur += " " + w
        else:
            chunks.append(cur)
            cur = w
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """Split into synthesis chunks: sentences first, then sub-split any sentence
    longer than max_chars on clause boundaries (commas etc.), packing clauses
    greedily; hard-split on words only if a single clause is still too long.
    Keeps long sentences from hitting the token cap / rambling; short ones intact."""
    out = []
    for sent in split_sentences(text):
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        cur = ""
        for clause in (c.strip() for c in _CLAUSE_SPLIT.split(sent) if c.strip()):
            if len(clause) > max_chars:
                if cur:
                    out.append(cur)
                    cur = ""
                out.extend(_split_words(clause, max_chars))
            elif not cur:
                cur = clause
            elif len(cur) + 1 + len(clause) <= max_chars:
                cur += " " + clause
            else:
                out.append(cur)
                cur = clause
        if cur:
            out.append(cur)
    return out
