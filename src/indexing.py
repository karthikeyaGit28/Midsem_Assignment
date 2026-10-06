"""
LegalLens Inverted Index Module
Implements classical Information Retrieval indexing structures:
- Dictionary + Postings lists with term frequencies and positions
- Positional index for exact phrase matching
- Document frequency (df) and inverse document frequency (idf)
- Document length tracking for BM25 and vector space normalization
- Zone indexing (case_name zone vs. text body zone)
"""

import math
import os
import sys
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import tokenize


class InvertedIndex:
    """
    Core Inverted Index structure mapping terms to postings with term frequency,
    positions, and zone metadata.
    """
    def __init__(self):
        # dictionary: term -> list of [doc_id, tf, [positions]]
        self.postings: Dict[str, List[Tuple[str, int, List[int]]]] = defaultdict(list)
        # document frequency: term -> number of docs containing term
        self.doc_freq: Dict[str, int] = defaultdict(int)
        # document length in tokens
        self.doc_lens: Dict[str, int] = {}
        # metadata storage: doc_id -> doc metadata
        self.docs_metadata: Dict[str, Dict[str, Any]] = {}
        # zone indexes: case_name index for high-precision entity matches
        self.case_name_postings: Dict[str, List[str]] = defaultdict(list)
        # total documents
        self.num_docs: int = 0
        # average document length
        self.avg_doc_len: float = 0.0

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build the inverted index from a list of document dicts."""
        self.num_docs = len(documents)
        total_tokens = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            self.docs_metadata[doc_id] = {
                "doc_id": doc_id,
                "case_name": doc["case_name"],
                "date": doc["date"],
                "court": doc.get("court", "Supreme Court of India"),
                "category": doc.get("category", "General Law"),
                "passages": doc.get("passages", []),
                "text_snippet": doc["text"][:300] + "..."
            }

            # Zone 1: Case Name
            case_tokens = tokenize(doc["case_name"])
            for t in set(case_tokens):
                self.case_name_postings[t].append(doc_id)

            # Zone 2: Main Document Text
            tokens = tokenize(doc["text"])
            doc_len = len(tokens)
            self.doc_lens[doc_id] = doc_len
            total_tokens += doc_len

            # Track positions and frequencies
            term_positions = defaultdict(list)
            for pos, tok in enumerate(tokens):
                term_positions[tok].append(pos)

            for term, positions in term_positions.items():
                self.postings[term].append((doc_id, len(positions), positions))
                self.doc_freq[term] += 1

        self.avg_doc_len = total_tokens / self.num_docs if self.num_docs > 0 else 0.0
        print(f"Indexed {self.num_docs} documents, vocabulary size: {len(self.postings)} terms, avg doc length: {self.avg_doc_len:.1f} tokens.")

    def get_idf(self, term: str) -> float:
        """Calculate Robertson-Spärck Jones IDF standard for BM25."""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # Standard BM25 IDF formulation: log((N - df + 0.5) / (df + 0.5) + 1.0)
        return math.log(((self.num_docs - df + 0.5) / (df + 0.5)) + 1.0)

    def get_standard_idf(self, term: str) -> float:
        """Standard log(N / df) for TF-IDF."""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.num_docs + 1.0) / (df + 1.0)) + 1.0

    def phrase_query(self, phrase: str) -> List[str]:
        """Positional index lookup to find exact phrase matches across documents."""
        phrase_tokens = tokenize(phrase)
        if not phrase_tokens:
            return []
        if len(phrase_tokens) == 1:
            term = phrase_tokens[0]
            return [doc_id for doc_id, _, _ in self.postings.get(term, [])]

        # Intersect positional postings
        first_term = phrase_tokens[0]
        first_postings = {doc_id: pos_list for doc_id, _, pos_list in self.postings.get(first_term, [])}
        matching_docs = []

        for doc_id, initial_positions in first_postings.items():
            candidates = set(initial_positions)
            for offset, term in enumerate(phrase_tokens[1:], start=1):
                doc_term_posts = [pos for d, _, pos in self.postings.get(term, []) if d == doc_id]
                if not doc_term_posts:
                    candidates = set()
                    break
                term_positions = set(doc_term_posts[0])
                # Check for consecutive positions
                candidates = {c + 1 for c in candidates if (c + 1) in term_positions}
                if not candidates:
                    break
            if candidates:
                matching_docs.append(doc_id)

        return matching_docs

    def save(self, file_path: str):
        """Save indexed structure to binary disk file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Index successfully saved to {file_path}")

    @classmethod
    def load(cls, file_path: str) -> 'InvertedIndex':
        """Load indexed structure from binary disk file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


def build_and_save_index(docs_path: str = "data/processed/documents.json", index_path: str = "data/processed/inverted_index.pkl"):
    import json
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    index = InvertedIndex()
    index.build_index(docs)
    index.save(index_path)
    return index


if __name__ == "__main__":
    build_and_save_index()
