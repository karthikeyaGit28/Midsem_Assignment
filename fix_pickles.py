import sys
import os
import pickle

sys.path.insert(0, os.path.abspath("."))

from src.indexing import InvertedIndex, build_and_save_index
from src.bm25_retriever import BM25Retriever, build_and_save_bm25
from src.tfidf_retriever import TFIDFRetriever, build_and_save_tfidf

# Ensure __module__ is set to the package name, not __main__
InvertedIndex.__module__ = "src.indexing"
BM25Retriever.__module__ = "src.bm25_retriever"
TFIDFRetriever.__module__ = "src.tfidf_retriever"

print("Rebuilding inverted index and BM25 retriever with explicit module names...")
build_and_save_index("data/processed/documents.json", "data/processed/inverted_index.pkl")
build_and_save_bm25("data/processed/documents.json", "data/processed/bm25_retriever.pkl")
build_and_save_tfidf("data/processed/documents.json", "data/processed/tfidf_vectorizer.pkl")

print("Testing unpickling...")
with open("data/processed/bm25_retriever.pkl", "rb") as f:
    bm25 = pickle.load(f)
print("BM25 unpickled successfully! Type:", type(bm25))

with open("data/processed/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
print("TF-IDF unpickled successfully! Type:", type(tfidf))
