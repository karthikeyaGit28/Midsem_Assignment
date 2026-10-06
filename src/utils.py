"""
LegalLens Utility Functions
Provides helper routines for I/O, scoring normalization, and result formatting.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple


def load_json(filepath: str) -> Any:
    """Load JSON data from a given path."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def min_max_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Min-Max normalization mapping raw scores to [0.0, 1.0].
    Handles flat/identical score arrays cleanly.
    """
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    diff = max_val - min_val
    if diff < eps:
        # If all scores are equal, return zeros or ones
        return np.ones_like(scores) if max_val > 0 else np.zeros_like(scores)
    return (scores - min_val) / diff


def format_search_result(
    doc_id: str,
    metadata: Dict[str, Any],
    score: float,
    lexical_score: float = 0.0,
    semantic_score: float = 0.0,
    best_passage: str = "",
    matched_terms: List[str] = None
) -> Dict[str, Any]:
    """Uniform result dictionary schema for CLI, UI, and evaluations."""
    return {
        "doc_id": doc_id,
        "case_name": metadata.get("case_name", "Unknown Case"),
        "date": metadata.get("date", "Unknown Date"),
        "court": metadata.get("court", "Supreme Court of India"),
        "category": metadata.get("category", "General Law"),
        "final_score": float(score),
        "bm25_score": float(lexical_score),
        "semantic_score": float(semantic_score),
        "passage": best_passage,
        "matched_terms": matched_terms or []
    }
