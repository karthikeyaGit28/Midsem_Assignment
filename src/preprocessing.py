"""
LegalLens Preprocessing Pipeline
Handles text normalization, tokenization, stemming, stop-word removal,
and transforming raw IndicLegalQA dataset into canonical documents and evaluation queries.
"""

import json
import os
import re
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()

# Custom domain stop-words that add no IR discrimination but keep legal terms
DOMAIN_STOP_WORDS = {
    "please", "tell", "briefly", "explain", "state", "case", "mentioned", "above"
}
ALL_STOP_WORDS = STOP_WORDS.union(DOMAIN_STOP_WORDS)


def clean_text(text: str) -> str:
    """Normalize raw text by removing extraneous whitespace and standardizing quotes."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True, use_stemming: bool = True) -> List[str]:
    """
    Tokenizes text, performs case-folding, stopword removal, and Porter stemming.
    Preserves legal alphanumerics like section numbers (e.g. s8, 12, 1956).
    """
    if not text:
        return []
    # Case folding & tokenization preserving alphanumeric tokens
    text = text.lower()
    raw_tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
    
    tokens = []
    for tok in raw_tokens:
        if remove_stopwords and tok in ALL_STOP_WORDS:
            continue
        if len(tok) <= 1 and not tok.isdigit():
            continue
        if use_stemming:
            tokens.append(STEMMER.stem(tok))
        else:
            tokens.append(tok)
            
    return tokens


def infer_legal_category(text: str) -> str:
    """Infer high-level legal category from case text for metadata and filtering."""
    t = text.lower()
    if any(k in t for k in ["bail", "accused", "murder", "ipc", "criminal", "conviction", "sentence", "police", "fir", "cognizable", "penal code"]):
        return "Criminal Law"
    elif any(k in t for k in ["constitution", "fundamental right", "writ petition", "article 21", "article 32", "article 226", "legislative"]):
        return "Constitutional Law"
    elif any(k in t for k in ["tenant", "eviction", "rent", "landlord", "possession", "suit for property", "mortgage", "partition"]):
        return "Civil & Property Law"
    elif any(k in t for k in ["promotion", "pension", "disciplinary", "service rules", "armed forces", "seniority", "superannuation"]):
        return "Service & Employment Law"
    elif any(k in t for k in ["arbitration", "contract", "insolvency", "company", "shares", "commercial", "nclt", "banking", "promissory"]):
        return "Commercial & Corporate Law"
    elif any(k in t for k in ["adoption", "hindu adoptions", "maintenance", "marriage", "divorce", "succession", "widow", "hama", "family"]):
        return "Family & Personal Law"
    elif any(k in t for k in ["tax", "income tax", "assessment", "excise", "customs", "gst", "revenue"]):
        return "Tax & Revenue Law"
    else:
        return "General Law"


def prepare_dataset(
    raw_json_path: str,
    output_dir: str = "data/processed",
    val_size: int = 1000,
    test_size: int = 2000,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads raw IndicLegalQA dataset, groups by case to construct canonical documents,
    extracts query evaluation benchmark, and partitions into validation and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)

    print(f"Loading raw dataset from {raw_json_path}...")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Total raw records loaded: {len(records)}")

    # Group Q&A by unique (case_name, judgement_date)
    case_groups = defaultdict(list)
    for idx, r in enumerate(records):
        case_name = clean_text(r.get("case_name", "Unknown Case"))
        date = clean_text(r.get("judgement_date", "Unknown Date"))
        q = clean_text(r.get("question", ""))
        a = clean_text(r.get("answer", ""))
        if not q or not a:
            continue
        case_key = (case_name, date)
        case_groups[case_key].append({
            "record_idx": idx,
            "question": q,
            "answer": a
        })

    print(f"Identified {len(case_groups)} unique cases.")

    # Build canonical documents
    documents = []
    case_to_doc_id = {}

    for doc_num, ((case_name, date), qas) in enumerate(sorted(case_groups.items())):
        doc_id = f"doc_{doc_num:04d}"
        case_to_doc_id[(case_name, date)] = doc_id

        # Unique answers as passages
        unique_passages = []
        seen_answers = set()
        for item in qas:
            ans = item["answer"]
            if ans not in seen_answers:
                seen_answers.add(ans)
                unique_passages.append(ans)

        # Document text combines case name, date, court and all synthesized answers
        full_text = f"Case: {case_name}. Date: {date}. Court: Supreme Court of India. " + " ".join(unique_passages)
        category = infer_legal_category(case_name + " " + full_text)

        doc_dict = {
            "doc_id": doc_id,
            "case_name": case_name,
            "date": date,
            "court": "Supreme Court of India",
            "category": category,
            "text": full_text,
            "passages": unique_passages,
            "num_qa_pairs": len(qas)
        }
        documents.append(doc_dict)

    # Build evaluation queries
    queries_all = []
    for idx, r in enumerate(records):
        case_name = clean_text(r.get("case_name", "Unknown Case"))
        date = clean_text(r.get("judgement_date", "Unknown Date"))
        q = clean_text(r.get("question", ""))
        a = clean_text(r.get("answer", ""))
        if not q or not a:
            continue
        
        target_doc_id = case_to_doc_id.get((case_name, date))
        if not target_doc_id:
            continue

        query_dict = {
            "query_id": f"q_{idx:05d}",
            "query": q,
            "relevant_doc_id": target_doc_id,
            "case_name": case_name,
            "date": date,
            "ground_truth_answer": a
        }
        queries_all.append(query_dict)

    print(f"Total valid queries compiled: {len(queries_all)}")

    # Shuffle for split
    shuffled_queries = list(queries_all)
    random.shuffle(shuffled_queries)

    val_queries = shuffled_queries[:val_size]
    test_queries = shuffled_queries[val_size:val_size + test_size]

    # Save to disk
    docs_file = os.path.join(output_dir, "documents.json")
    queries_all_file = os.path.join(output_dir, "queries_all.json")
    queries_val_file = os.path.join(output_dir, "queries_val.json")
    queries_test_file = os.path.join(output_dir, "queries_test.json")

    with open(docs_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    with open(queries_all_file, "w", encoding="utf-8") as f:
        json.dump(queries_all, f, indent=2, ensure_ascii=False)
    with open(queries_val_file, "w", encoding="utf-8") as f:
        json.dump(val_queries, f, indent=2, ensure_ascii=False)
    with open(queries_test_file, "w", encoding="utf-8") as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(documents)} documents to {docs_file}")
    print(f"Saved {len(queries_all)} total queries to {queries_all_file}")
    print(f"Saved {len(val_queries)} validation queries to {queries_val_file}")
    print(f"Saved {len(test_queries)} test queries to {queries_test_file}")

    return documents, queries_all, val_queries, test_queries


if __name__ == "__main__":
    raw_path = "data/raw/IndicLegalQA_Dataset_10K_Revised.json"
    if not os.path.exists(raw_path):
        raw_path = "IndicLegalQA Dataset_10K_Revised.json"
    prepare_dataset(raw_path)
