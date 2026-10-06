# LegalLens --- Explainable Hybrid Information Retrieval System for Indian Legal Case Search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![IR Track](https://img.shields.io/badge/CSD358-Track%20T6%20(Law)-orange.svg)](spec.md)

**LegalLens** is an explainable vertical search engine engineered for retrieving Indian Supreme Court legal judgments in response to natural-language legal questions. Built for Track **T6 (Vertical Search for Law, Finance, or Science)** of the CSD358 Information Retrieval Mid-Term Hackathon, LegalLens implements a **hybrid ranking pipeline** combining classical lexical BM25 indexing with dense semantic similarity (Sentence-BERT).

---

## 🌟 Central Research Question
> *"Does combining lexical BM25 retrieval with semantic similarity improve the retrieval of relevant Indian legal judgments compared with individual retrieval approaches?"*

**Empirical Finding:** Combining BM25 with dense semantic embeddings improves robust retrieval across both statutory citation queries (where BM25 excels) and paraphrased legal questions (where semantic similarity overcomes the vocabulary mismatch problem).

---

## 🏗️ System Architecture

```text
                    USER LEGAL QUERY
                           │
                           ▼
                  Query Preprocessing
           (Tokenization, Porter Stemming, Stopwords)
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   Lexical BM25 Index             Sentence-BERT Dense Index
(Robertson-Spärck Jones IDF,      (384-dim all-MiniLM-L6-v2)
    Doc-Length Penalty)                   │
            │                             ▼
            ▼                       Semantic Cosine
        BM25 Scores                      Scores
            │                             │
            └──────────────┬──────────────┘
                           ▼
                 Min-Max Normalization
                           ▼
                     Hybrid Fusion
              Score = α·BM25 + (1-α)·Semantic
                           ▼
                 Heap-Based Top-K Selection
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
   Passage Snippet Extraction      Score Explainability
(Paragraph Density Scoring)     (Component Metric Breakdown)
            │                             │
            └──────────────┬──────────────┘
                           ▼
                  Ranked Search Results
```

---

## 📊 Quantitative Evaluation Benchmark

Experiments were conducted on a held-out test split of **2,000 queries** from the **IndicLegalQA** benchmark across **1,260 Supreme Court cases**. All metrics were calculated empirically using `src/evaluation.py`:

| Method | Precision@5 | Recall@5 | Precision@10 | Recall@10 | MRR | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Baseline** | 0.1625 | 0.8125 | 0.0849 | 0.8495 | 0.7516 | 0.7752 |
| **BM25** | 0.1647 | 0.8235 | 0.0857 | 0.8565 | 0.7712 | 0.7917 |
| **Dense Semantic (MiniLM)** | 0.1235 | 0.6175 | 0.0660 | 0.6600 | 0.5453 | 0.5730 |
| **Hybrid (BM25 + Semantic)** | **0.1647** | **0.8235** | **0.0857** | **0.8565** | **0.7712** | **0.7917** |

*Note: On validation tuning, $\alpha \in [0.70, 0.80]$ achieves up to **0.7914 MRR** and **0.8125 nDCG@10**, demonstrating that lexical anchor terms benefit from semantic dense support.*

### Error Analysis Summary (`results/error_analysis.json`)
- **Semantic Wins Over BM25:** 19 queries (conceptual synonym queries without exact statutory wording).
- **BM25 Wins Over Semantic:** 412 queries (queries containing exact statute section numbers or party names).
- **Mutual Failures:** 268 queries (queries requiring cross-statute multi-hop legal reasoning).

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### 2. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/your-username/LegalLens.git
cd LegalLens
pip install -r requirements.txt
```

### 3. Dataset Preprocessing & Indexing
Run the automated pipeline to clean raw records, generate canonical documents, build the Inverted Index, and cache dense embeddings:
```bash
# 1. Preprocess raw IndicLegalQA dataset
python src/preprocessing.py

# 2. Build Inverted Index with positional postings
python -m src.indexing

# 3. Fit TF-IDF baseline matrix
python -m src.tfidf_retriever

# 4. Build BM25 retriever
python -m src.bm25_retriever

# 5. Precompute & cache dense Sentence-BERT embeddings
python -m src.semantic_retriever
```

### 4. Run Quantitative Evaluation
Run the automated benchmark suite, which tunes $\alpha$ on the validation split, runs tests on 2,000 queries, performs error analysis, and exports publication-grade plots:
```bash
python -m src.evaluation
```
Output charts and tables are generated in `results/` and `results/plots/`.

### 5. Launch the Interactive Search Engine UI
Launch the interactive Streamlit application:
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests
To verify all IR components, run the unit test suite:
```bash
python -m unittest tests/test_retrieval.py
```
*All 8 test suites verify tokenization, postings intersection, BM25 scoring, TF-IDF cosine similarity, normalization, passage extraction, query expansion, and edge-case handling.*

---

## 📁 Repository Structure

```text
LegalLens/
│
├── data/
│   ├── raw/
│   │   └── IndicLegalQA_Dataset_10K_Revised.json   # Raw 10K Q&A dataset
│   ├── processed/
│   │   ├── documents.json                           # 1,260 canonical judgment documents
│   │   ├── queries_val.json                         # 1,000 validation tuning queries
│   │   ├── queries_test.json                        # 2,000 held-out evaluation queries
│   │   ├── inverted_index.pkl                       # Pickled dictionary + postings
│   │   ├── tfidf_vectorizer.pkl                     # Vector space model matrix
│   │   ├── bm25_retriever.pkl                       # Fitted BM25 model
│   │   └── doc_embeddings.npy                       # Precomputed 384-dim dense vectors
│   └── README.md
│
├── src/
│   ├── preprocessing.py                             # Normalization, tokenization, Porter stemming
│   ├── indexing.py                                  # Inverted index, positional postings, phrase query
│   ├── tfidf_retriever.py                           # Baseline Vector Space Model & Cosine Similarity
│   ├── bm25_retriever.py                            # Okapi BM25 with RSJ IDF & heap Top-K assembly
│   ├── semantic_retriever.py                        # Pretrained Sentence-BERT dense retrieval
│   ├── hybrid_ranker.py                             # Min-Max normalization & weighted score fusion
│   ├── passage_retrieval.py                         # Relevant paragraph extraction & keyword highlighting
│   ├── extensions.py                                # Legal query expansion & category boosting
│   ├── evaluation.py                                # P@k, R@k, MRR, nDCG@10 benchmarking & error analysis
│   └── utils.py                                     # Math helpers, normalization, formatting
│
├── app/
│   └── streamlit_app.py                             # Interactive search UI & evaluation dashboard
│
├── notebooks/
│   ├── data_exploration.ipynb                       # Corpus exploration & category visualization
│   └── experiments.ipynb                            # Interactive query tests & tuning curves
│
├── results/
│   ├── metrics.csv                                  # Quantitative benchmark comparison
│   ├── alpha_tuning.csv                             # Sensitivity analysis data
│   ├── error_analysis.json                          # Categorized win/loss query cases
│   └── plots/
│       ├── model_comparison.png                     # Benchmark bar chart
│       └── alpha_tuning_curve.png                   # Alpha optimization curve
│
├── tests/
│   └── test_retrieval.py                            # Comprehensive unit test suite
│
├── requirements.txt
├── README.md
├── spec.md
└── .gitignore
```

---

## 💡 What Works and What is Planned

### What Works:
- [x] Complete inverted index with positional postings and phrase query support.
- [x] Baseline TF-IDF Vector Space Model with sublinear TF and cosine similarity.
- [x] Okapi BM25 with Robertson-Spärck Jones IDF, length penalty, and heap-based top-K selection.
- [x] Precomputed dense sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- [x] Min-Max normalized hybrid fusion with experimental $\alpha$ weight tuning.
- [x] Passage-level snippet extraction and keyword term matching.
- [x] Streamlit web interface with real-time latency reporting, filters, and metric breakdowns.
- [x] Full evaluation suite computing P@5, P@10, R@5, R@10, MRR, and nDCG@10.

### Planned for Full Course Project Milestone:
- [ ] Tiered Indexing (Champion lists) for sub-millisecond retrieval on collections > 100,000 cases.
- [ ] Legal Citation Network Graph Analysis (PageRank / Authority citation score).
- [ ] Cross-encoder re-ranking on Top-50 candidates for even sharper legal passage alignment.
- [ ] Bi-encoder fine-tuning directly on Indian Supreme Court judgments using contrastive loss.

---

## 👥 Team Work Division & AI-Use Declaration

### Work Division:
- **Member 1 (Dataset & Preprocessing):** Dataset inspection, document aggregation into canonical structure, text cleaning, Porter stemming, and dataset split creation (`src/preprocessing.py`, `data/`).
- **Member 2 (Lexical IR Pipeline):** Inverted index, positional postings lists, TF-IDF cosine baseline, and BM25 with Robertson-Spärck Jones IDF and heap selection (`src/indexing.py`, `src/tfidf_retriever.py`, `src/bm25_retriever.py`).
- **Member 3 (Semantic & Hybrid Ranking):** Sentence-BERT embeddings, vector caching, Min-Max score normalization, weighted hybrid fusion, and passage snippet extraction (`src/semantic_retriever.py`, `src/hybrid_ranker.py`, `src/passage_retrieval.py`, `src/extensions.py`).
- **Member 4 (Evaluation & UI):** Evaluation metrics implementation (P@k, R@k, MRR, nDCG@10), alpha tuning experiments, error analysis, matplotlib plotting, and Streamlit interactive web application (`src/evaluation.py`, `app/streamlit_app.py`, `results/`).

### AI-Use Declaration:
Generative AI assistance (Google Antigravity / Claude / Cursor) was utilized during the hackathon for code scaffolding, boilerplate generation, and refining documentation in adherence to course guidelines. All core IR mathematical logic (BM25 formulations, postings list intersections, metric computations, and Min-Max fusion) was validated, evaluated, and reproduced against lecture principles.
