"""
LegalLens Passage Retrieval & Snippet Extraction Module
Extracts the most relevant paragraph/passage from retrieved judgments (spec.md Section 6 FR8).
Computes passage-level scoring to pinpoint exact statutory interpretation or judicial holding.
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple
from src.preprocessing import tokenize


def extract_best_passage(
    query: str,
    passages: List[str],
    doc_text: str = "",
    max_words: int = 120
) -> Tuple[str, float]:
    """
    Given a query and a list of document passages, identify and return the passage
    with the highest term overlap / BM25-like density.
    If no explicit passages are provided, breaks doc_text into sentences/paragraphs.
    """
    candidate_passages = []
    if passages and len(passages) > 0:
        candidate_passages = [p.strip() for p in passages if len(p.strip()) > 20]

    if not candidate_passages and doc_text:
        # Fallback: split by sentence or paragraph
        raw_splits = re.split(r'(?<=[.!?])\s+', doc_text)
        chunk = []
        c_len = 0
        for s in raw_splits:
            words = s.split()
            if c_len + len(words) > 60:
                if chunk:
                    candidate_passages.append(" ".join(chunk))
                chunk = words
                c_len = len(words)
            else:
                chunk.extend(words)
                c_len += len(words)
        if chunk:
            candidate_passages.append(" ".join(chunk))

    if not candidate_passages:
        return doc_text[:300] + "...", 0.0

    q_tokens = set(tokenize(query))
    if not q_tokens:
        return candidate_passages[0], 0.0

    best_passage = candidate_passages[0]
    best_score = -1.0

    for p in candidate_passages:
        p_tokens = tokenize(p)
        if not p_tokens:
            continue
        # Term overlap with length penalty
        overlap = sum(1 for t in p_tokens if t in q_tokens)
        score = overlap / (len(p_tokens) ** 0.5 + 1e-5)
        if score > best_score:
            best_score = score
            best_passage = p

    # Truncate if passage is excessively long
    words = best_passage.split()
    if len(words) > max_words:
        best_passage = " ".join(words[:max_words]) + "..."

    return best_passage, float(best_score)


def highlight_matched_terms(text: str, query: str) -> str:
    """Highlight matched query terms using HTML <mark> tags for UI display."""
    q_tokens = tokenize(query, use_stemming=False)
    highlighted = text
    for term in set(q_tokens):
        if len(term) < 3:
            continue
        pattern = re.compile(rf'\b({re.escape(term)})\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark style="background-color: #ffeb3b; padding: 0 2px; border-radius: 2px;">\1</mark>', highlighted)
    return highlighted
