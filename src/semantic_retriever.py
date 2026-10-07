"""
LegalLens Semantic Retriever
Implements Dense Vector Retrieval using pretrained Sentence Transformers (spec.md Section 7.3).
Precomputes and caches document embeddings for real-time sub-second query retrieval.
"""

import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_json, format_search_result


class SemanticRetriever:
    """Dense Semantic Retriever using Sentence-Transformers and Cosine Similarity."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embeddings_path: str = "data/processed/doc_embeddings.npy",
        metadata_path: str = "data/processed/semantic_metadata.pkl"
    ):
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.model = None
        self.doc_ids: List[str] = []
        self.doc_embeddings: np.ndarray = None
        self.docs_metadata: Dict[str, Dict[str, Any]] = {}

    def _load_model(self):
        """Lazy load sentence transformer model onto CPU/GPU."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)

    def fit_and_encode(self, documents: List[Dict[str, Any]], batch_size: int = 64):
        """Generate and cache dense embeddings for all documents in collection."""
        self._load_model()
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

        print(f"Encoding {len(texts)} legal documents into dense vectors...")
        # Normalizing embeddings ensures dot product equals cosine similarity
        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print(f"Encoded documents shape: {self.doc_embeddings.shape}")
        self.save()

    def save(self):
        """Persist embeddings matrix and metadata to disk."""
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        np.save(self.embeddings_path, self.doc_embeddings)
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "doc_ids": self.doc_ids,
                "docs_metadata": self.docs_metadata,
                "model_name": self.model_name
            }, f)
        print(f"Saved dense embeddings to {self.embeddings_path} and metadata to {self.metadata_path}")

    @classmethod
    def load(
        cls,
        embeddings_path: str = "data/processed/doc_embeddings.npy",
        metadata_path: str = "data/processed/semantic_metadata.pkl"
    ) -> 'SemanticRetriever':
        """Load precomputed embeddings and metadata from disk."""
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        
        instance = cls(
            model_name=data.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            embeddings_path=embeddings_path,
            metadata_path=metadata_path
        )
        instance.doc_ids = data["doc_ids"]
        instance.docs_metadata = data["docs_metadata"]
        instance.doc_embeddings = np.load(embeddings_path)
        return instance

    def encode_query(self, query: str) -> np.ndarray:
        """Encode user query into a normalized dense vector."""
        self._load_model()
        return self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    def encode_queries_batch(self, queries: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode batch of queries for fast bulk evaluation."""
        self._load_model()
        return self.model.encode(queries, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True)

    def get_all_scores(self, query: str) -> np.ndarray:
        """Compute cosine similarity of query vector against all document vectors."""
        q_emb = self.encode_query(query)
        # Cosine similarity is dot product when vectors are L2-normalized
        return np.dot(self.doc_embeddings, q_emb)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve top_k documents by semantic cosine similarity."""
        scores = self.get_all_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            doc_id = self.doc_ids[idx]
            meta = self.docs_metadata[doc_id]

            results.append(format_search_result(
                doc_id=doc_id,
                metadata=meta,
                score=score,
                lexical_score=0.0,
                semantic_score=score,
                best_passage=meta.get("passages", [""])[0] if meta.get("passages") else "",
                matched_terms=[]
            ))
        return results


def build_and_save_semantic(
    docs_path: str = "data/processed/documents.json",
    embeddings_path: str = "data/processed/doc_embeddings.npy",
    metadata_path: str = "data/processed/semantic_metadata.pkl"
) -> SemanticRetriever:
    documents = load_json(docs_path)
    retriever = SemanticRetriever(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path
    )
    retriever.fit_and_encode(documents)
    return retriever


if __name__ == "__main__":
    retriever = build_and_save_semantic()
    test_q = "Can a tenant be evicted without proper notice?"
    results = retriever.search(test_q, top_k=3)
    print(f"\nTest Semantic Query: '{test_q}'")
    for r in results:
        print(f"[{r['semantic_score']:.4f}] {r['case_name']} ({r['category']})")
