"""
LegalLens Hybrid Ranker
Combines lexical BM25 retrieval with dense semantic similarity (spec.md Section 7.4 & FR5).
Features Min-Max score normalization, configurable alpha weighting, explainable score breakdown,
and snippet passage extraction.
"""

import os
import sys
import heapq
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.tfidf_retriever import TFIDFRetriever
from src.passage_retrieval import extract_best_passage
from src.preprocessing import tokenize
from src.utils import min_max_normalize, format_search_result


class HybridRanker:
    """
    Hybrid Information Retrieval ranker fusing lexical BM25 and dense semantic similarity:
    FinalScore = alpha * NormalizedBM25 + (1 - alpha) * NormalizedSemantic
    """
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        alpha: float = 0.5
    ):
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.alpha = float(alpha)
        # Ensure document ID alignment
        assert self.bm25.doc_ids == self.semantic.doc_ids, "Document IDs must match between lexical and semantic models!"
        self.doc_ids = self.bm25.doc_ids
        self.docs_metadata = self.bm25.index.docs_metadata

    def set_alpha(self, alpha: float):
        """Set the lexical vs semantic fusion parameter alpha (0.0 <= alpha <= 1.0)."""
        self.alpha = max(0.0, min(1.0, float(alpha)))

    def rank_query(self, query: str, top_k: int = 10, alpha: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Rank documents for a natural-language legal query using normalized hybrid fusion.
        Returns Top-K cases with detailed ranking explanations and relevant passages.
        """
        w_bm25 = self.alpha if alpha is None else max(0.0, min(1.0, float(alpha)))
        w_sem = 1.0 - w_bm25

        # 1. Compute lexical BM25 scores
        raw_bm25 = self.bm25.get_all_scores(query)

        # 2. Compute semantic dense scores
        raw_sem = self.semantic.get_all_scores(query)

        # 3. Score Normalization (spec.md Section 7.4)
        norm_bm25 = min_max_normalize(raw_bm25)
        norm_sem = min_max_normalize(raw_sem)

        # 4. Hybrid score combination
        final_scores = (w_bm25 * norm_bm25) + (w_sem * norm_sem)

        # 5. Top-K selection via heap
        top_indices = heapq.nlargest(top_k, range(len(final_scores)), key=lambda i: final_scores[i])

        query_terms = set(tokenize(query))
        results = []

        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            meta = self.docs_metadata[doc_id]
            f_score = float(final_scores[idx])

            # Extract relevant passage (FR8)
            passages = meta.get("passages", [])
            best_passage, _ = extract_best_passage(query, passages)

            # Extract matched terms
            doc_terms = set(tokenize(meta.get("case_name", "") + " " + meta.get("text_snippet", "")))
            matched = list(query_terms.intersection(doc_terms))

            res = {
                "doc_id": doc_id,
                "case_name": meta.get("case_name", "Unknown Case"),
                "date": meta.get("date", "Unknown Date"),
                "court": meta.get("court", "Supreme Court of India"),
                "category": meta.get("category", "General Law"),
                "final_score": round(f_score, 4),
                "bm25_raw": round(float(raw_bm25[idx]), 4),
                "bm25_norm": round(float(norm_bm25[idx]), 4),
                "semantic_raw": round(float(raw_sem[idx]), 4),
                "semantic_norm": round(float(norm_sem[idx]), 4),
                "alpha": w_bm25,
                "passage": best_passage,
                "matched_terms": matched,
                "full_text": meta.get("text_snippet", "")
            }
            results.append(res)

        return results

    def batch_rank(
        self,
        queries: List[str],
        alpha: Optional[float] = None,
        top_k: int = 10
    ) -> List[List[str]]:
        """
        Fast vectorized batch ranking for quantitative evaluation.
        Returns List of Top-K doc_ids for each query.
        """
        w_bm25 = self.alpha if alpha is None else max(0.0, min(1.0, float(alpha)))
        w_sem = 1.0 - w_bm25

        # Semantic batch embeddings
        q_embs = self.semantic.encode_queries_batch(queries, batch_size=128)
        # Shape: (num_queries, num_docs)
        all_sem_scores = np.dot(q_embs, self.semantic.doc_embeddings.T)

        all_ranked_doc_ids = []

        for q_idx, q in enumerate(queries):
            raw_bm25 = self.bm25.get_all_scores(q)
            raw_sem = all_sem_scores[q_idx]

            norm_bm25 = min_max_normalize(raw_bm25)
            norm_sem = min_max_normalize(raw_sem)

            final_scores = (w_bm25 * norm_bm25) + (w_sem * norm_sem)
            top_indices = np.argsort(final_scores)[::-1][:top_k]
            ranked_docs = [self.doc_ids[i] for i in top_indices]
            all_ranked_doc_ids.append(ranked_docs)

        return all_ranked_doc_ids


def load_hybrid_system() -> HybridRanker:
    """Convenience factory to load pretrained BM25 and Semantic models."""
    bm25 = BM25Retriever.load("data/processed/bm25_retriever.pkl")
    semantic = SemanticRetriever.load("data/processed/doc_embeddings.npy", "data/processed/semantic_metadata.pkl")
    return HybridRanker(bm25, semantic, alpha=0.5)


if __name__ == "__main__":
    ranker = load_hybrid_system()
    test_q = "Can a tenant be evicted without proper notice?"
    results = ranker.rank_query(test_q, top_k=3, alpha=0.5)
    print(f"\n--- Hybrid Retrieval Results for '{test_q}' (alpha=0.5) ---")
    for i, r in enumerate(results, start=1):
        print(f"\n{i}. {r['case_name']} (Final Score: {r['final_score']})")
        print(f"   Court: {r['court']} | Date: {r['date']} | Category: {r['category']}")
        print(f"   BM25 Norm: {r['bm25_norm']} | Semantic Norm: {r['semantic_norm']}")
        print(f"   Relevant Passage: \"{r['passage'][:150]}...\"")
