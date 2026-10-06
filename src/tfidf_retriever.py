"""
LegalLens TF-IDF Retriever (Baseline)
Implements Vector Space Model with TF-IDF weighting and Cosine Similarity (spec.md Section 7.1).
Serves as the foundational Information Retrieval baseline.
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import tokenize, clean_text
from src.utils import load_json, format_search_result


class TFIDFRetriever:
    """TF-IDF Vector Space Retriever using Cosine Similarity."""
    
    def __init__(self, vectorizer_path: str = "data/processed/tfidf_vectorizer.pkl"):
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,  # Use custom tokenizer
            lowercase=True,
            norm='l2',
            sublinear_tf=True
        )
        self.doc_ids: List[str] = []
        self.doc_matrix = None
        self.docs_metadata: Dict[str, Dict[str, Any]] = {}

    def fit(self, documents: List[Dict[str, Any]]):
        """Fit TF-IDF matrix over all documents."""
        self.doc_ids = [doc["doc_id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        self.docs_metadata = {
            doc["doc_id"]: {
                "doc_id": doc["doc_id"],
                "case_name": doc["case_name"],
                "date": doc["date"],
                "court": doc.get("court", "Supreme Court of India"),
                "category": doc.get("category", "General Law"),
                "passages": doc.get("passages", []),
                "text_snippet": doc["text"][:300] + "..."
            }
            for doc in documents
        }

        print(f"Fitting TF-IDF Vectorizer on {len(texts)} documents...")
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix built with shape {self.doc_matrix.shape}")

    def save(self, filepath: str = None):
        """Persist retriever to disk."""
        path = filepath or self.vectorizer_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "doc_ids": self.doc_ids,
                "doc_matrix": self.doc_matrix,
                "docs_metadata": self.docs_metadata
            }, f)
        print(f"TF-IDF retriever saved to {path}")

    @classmethod
    def load(cls, filepath: str = "data/processed/tfidf_vectorizer.pkl") -> 'TFIDFRetriever':
        """Load fitted retriever from disk."""
        instance = cls(vectorizer_path=filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            instance.vectorizer = data["vectorizer"]
            instance.doc_ids = data["doc_ids"]
            instance.doc_matrix = data["doc_matrix"]
            instance.docs_metadata = data["docs_metadata"]
        return instance

    def get_all_scores(self, query: str) -> np.ndarray:
        """Calculate cosine similarity scores across all documents."""
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()
        return sims

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top_k documents by TF-IDF cosine similarity."""
        scores = self.get_all_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        q_tokens = set(tokenize(query))
        for idx in top_indices:
            score = float(scores[idx])
            doc_id = self.doc_ids[idx]
            meta = self.docs_metadata[doc_id]

            # Find matching terms
            doc_tokens = set(tokenize(meta.get("case_name", "") + " " + meta.get("text_snippet", "")))
            matched = list(q_tokens.intersection(doc_tokens))

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


def build_and_save_tfidf(docs_path: str = "data/processed/documents.json", save_path: str = "data/processed/tfidf_vectorizer.pkl"):
    documents = load_json(docs_path)
    retriever = TFIDFRetriever(vectorizer_path=save_path)
    retriever.fit(documents)
    retriever.save()
    return retriever


if __name__ == "__main__":
    retriever = build_and_save_tfidf()
    # Test query
    test_q = "Can a tenant be evicted without proper notice?"
    res = retriever.search(test_q, top_k=3)
    print(f"\nTest TF-IDF Query: '{test_q}'")
    for r in res:
        print(f"[{r['final_score']:.4f}] {r['case_name']} ({r['category']})")
