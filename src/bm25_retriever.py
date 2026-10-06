"""
LegalLens BM25 Retriever
Implements Okapi BM25 ranking algorithm based on Inverted Index postings and Robertson-Spärck Jones IDF.
Features term frequency saturation (k1), document length normalization (b), and heap-based Top-K selection.
(spec.md Section 7.2)
"""

import math
import os
import sys
import pickle
import heapq
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import tokenize
from src.indexing import InvertedIndex
from src.utils import load_json, format_search_result


class BM25Retriever:
    """
    Okapi BM25 implementation directly utilizing the Inverted Index structures.
    Uses lecture-aligned IR scoring:
    Score(D, Q) = sum_{q in Q} IDF(q) * [ tf(q, D) * (k1 + 1) ] / [ tf(q, D) + k1 * (1 - b + b * (|D| / avgdl)) ]
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, index: InvertedIndex = None):
        self.k1 = k1
        self.b = b
        self.index = index
        self.doc_ids: List[str] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        if index is not None:
            self._init_doc_mapping()

    def _init_doc_mapping(self):
        self.doc_ids = sorted(list(self.index.docs_metadata.keys()))
        self.doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def fit(self, documents: List[Dict[str, Any]]):
        """Build underlying Inverted Index from document collection."""
        self.index = InvertedIndex()
        self.index.build_index(documents)
        self._init_doc_mapping()

    def get_all_scores(self, query: str) -> np.ndarray:
        """
        Compute BM25 scores across all documents in the corpus for a given query.
        Returns a 1D NumPy array aligned with self.doc_ids.
        """
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        query_tokens = tokenize(query)
        if not query_tokens or self.index is None:
            return scores

        avgdl = self.index.avg_doc_len
        k1 = self.k1
        b = self.b

        # Process each unique query term and accumulate scores from inverted postings
        term_counts = defaultdict(int)
        for t in query_tokens:
            term_counts[t] += 1

        for term, qtf in term_counts.items():
            postings = self.index.postings.get(term, [])
            if not postings:
                continue
            idf = self.index.get_idf(term)
            if idf <= 0:
                continue

            # Case Name Zone Boost: if term matches case name directly, apply high-precision entity boost
            case_matches = set(self.index.case_name_postings.get(term, []))

            for doc_id, tf, _ in postings:
                idx = self.doc_id_to_idx.get(doc_id)
                if idx is None:
                    continue
                doc_len = self.index.doc_lens.get(doc_id, avgdl)
                norm_len = 1.0 - b + b * (doc_len / avgdl) if avgdl > 0 else 1.0
                tf_component = (tf * (k1 + 1.0)) / (tf + k1 * norm_len)
                
                term_score = idf * tf_component * qtf
                if doc_id in case_matches:
                    term_score *= 1.3  # Zone weight boost for exact entity match in title
                scores[idx] += term_score

        return scores

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute BM25 search using min-heap for efficient top-k selection (Scoring & Result Assembly).
        """
        scores = self.get_all_scores(query)
        if len(scores) == 0:
            return []

        # Heap-based Top-K selection (Lecture Syllabus Topic: Scoring and result assembly)
        top_k_indices = heapq.nlargest(top_k, range(len(scores)), key=lambda i: scores[i])

        query_terms = set(tokenize(query))
        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc_id = self.doc_ids[idx]
            meta = self.index.docs_metadata[doc_id]

            # Matching query terms
            doc_terms = set(tokenize(meta.get("case_name", "") + " " + meta.get("text_snippet", "")))
            matched = list(query_terms.intersection(doc_terms))

            results.append(format_search_result(
                doc_id=doc_id,
                metadata=meta,
                score=score,
                lexical_score=score,
                semantic_score=0.0,
                best_passage=meta.get("passages", [""])[0] if meta.get("passages") else "",
                matched_terms=matched
            ))

        return results

    def save(self, filepath: str = "data/processed/bm25_retriever.pkl"):
        """Save BM25 retriever instance."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"BM25 retriever saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "data/processed/bm25_retriever.pkl") -> 'BM25Retriever':
        """Load BM25 retriever instance."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


def build_and_save_bm25(docs_path: str = "data/processed/documents.json", save_path: str = "data/processed/bm25_retriever.pkl") -> BM25Retriever:
    documents = load_json(docs_path)
    retriever = BM25Retriever()
    retriever.fit(documents)
    retriever.save(save_path)
    return retriever


if __name__ == "__main__":
    retriever = build_and_save_bm25()
    test_q = "Can a tenant be evicted without proper notice?"
    results = retriever.search(test_q, top_k=3)
    print(f"\nTest BM25 Query: '{test_q}'")
    for r in results:
        print(f"[{r['bm25_score']:.4f}] {r['case_name']} ({r['category']}) - Matched: {r['matched_terms']}")
