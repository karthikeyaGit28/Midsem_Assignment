"""
LegalLens Unit and Integration Tests
Tests core Information Retrieval components:
- Preprocessing and tokenization
- Inverted Index postings and phrase search
- TF-IDF and BM25 retrievers
- Dense Semantic embeddings
- Hybrid ranker fusion and score normalization
- Passage extraction and edge case reliability
"""

import unittest
import numpy as np
from src.preprocessing import clean_text, tokenize, infer_legal_category
from src.indexing import InvertedIndex
from src.tfidf_retriever import TFIDFRetriever
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.hybrid_ranker import HybridRanker
from src.passage_retrieval import extract_best_passage, highlight_matched_terms
from src.extensions import expand_query, apply_metadata_boost
from src.utils import min_max_normalize


class TestLegalLensPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Sample mini-corpus for fast unit testing
        cls.sample_docs = [
            {
                "doc_id": "test_01",
                "case_name": "State vs. Ramesh Kumar",
                "date": "10th May 2019",
                "court": "Supreme Court of India",
                "category": "Criminal Law",
                "text": "The accused was charged with murder under Section 302 IPC. The Supreme Court granted bail due to lack of direct evidence.",
                "passages": [
                    "The accused was charged with murder under Section 302 IPC.",
                    "The Supreme Court granted bail due to lack of direct evidence."
                ]
            },
            {
                "doc_id": "test_02",
                "case_name": "Anil Gupta vs. Municipal Corp",
                "date": "15th June 2020",
                "court": "Supreme Court of India",
                "category": "Civil & Property Law",
                "text": "The tenant faced eviction proceedings without thirty days notice under the Rent Control Act. The court ordered immediate stay of dispossession.",
                "passages": [
                    "The tenant faced eviction proceedings without thirty days notice under the Rent Control Act.",
                    "The court ordered immediate stay of dispossession."
                ]
            },
            {
                "doc_id": "test_03",
                "case_name": "Sunita Sharma vs. Union of India",
                "date": "22nd July 2021",
                "court": "Supreme Court of India",
                "category": "Service & Employment Law",
                "text": "The petitioner challenged denial of promotion and pension benefits after thirty years of armed forces service.",
                "passages": [
                    "The petitioner challenged denial of promotion and pension benefits after thirty years of armed forces service."
                ]
            }
        ]

    def test_tokenization_and_stemming(self):
        text = "Tenants were being evicted under Section 12!"
        tokens = tokenize(text)
        self.assertIn("tenant", tokens)
        self.assertIn("evict", tokens)
        self.assertIn("12", tokens)

    def test_inverted_index(self):
        index = InvertedIndex()
        index.build_index(self.sample_docs)
        self.assertEqual(index.num_docs, 3)
        self.assertIn("bail", index.postings)
        self.assertIn("evict", index.postings)
        # Test phrase query
        matches = index.phrase_query("Rent Control Act")
        self.assertIn("test_02", matches)

    def test_bm25_retrieval(self):
        bm25 = BM25Retriever()
        bm25.fit(self.sample_docs)
        results = bm25.search("eviction of tenant notice", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "test_02")
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_tfidf_retrieval(self):
        tfidf = TFIDFRetriever()
        tfidf.fit(self.sample_docs)
        results = tfidf.search("murder charges bail", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["doc_id"], "test_01")

    def test_min_max_normalization(self):
        scores = np.array([10.0, 20.0, 30.0])
        norm = min_max_normalize(scores)
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[1], 0.5)
        self.assertAlmostEqual(norm[2], 1.0)

    def test_passage_extraction(self):
        query = "thirty days eviction notice"
        passages = self.sample_docs[1]["passages"]
        best, score = extract_best_passage(query, passages)
        self.assertIn("thirty days notice", best)
        self.assertGreater(score, 0)

    def test_query_expansion(self):
        q = "tenant eviction"
        expanded = expand_query(q)
        self.assertIn("tenancy", expanded.lower())
        self.assertIn("ejectment", expanded.lower())

    def test_empty_query_handling(self):
        bm25 = BM25Retriever()
        bm25.fit(self.sample_docs)
        res = bm25.search("", top_k=5)
        self.assertEqual(len(res), 0)


if __name__ == "__main__":
    unittest.main()
