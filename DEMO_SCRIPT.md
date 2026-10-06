# LegalLens --- Demo Video Presentation Script (5--8 Minutes)
**CSD358: Information Retrieval Hackathon --- Mid-Term Assignment**  
**Track:** T6 --- Vertical Search for Law, Finance or Science  
**Format:** Unlisted YouTube / Google Drive link | Live running system (No slides)

---

## ⏱️ Video Timeline & Speaker Breakdown

| Time | Section | Speaker / Role | Screen Content |
| :--- | :--- | :--- | :--- |
| **0:00 - 1:00** | 1. Problem & Track Relevance | **Member 1** | Terminal / Dataset & IDE |
| **1:00 - 2:00** | 2. Dataset & Lexical Preprocessing | **Member 1** | VS Code (`src/preprocessing.py`, `src/indexing.py`) |
| **2:00 - 3:15** | 3. Lexical IR: TF-IDF vs. BM25 | **Member 2** | Streamlit UI + Terminal Output |
| **3:15 - 4:30** | 4. Dense Semantic & Hybrid Ranking | **Member 3** | Streamlit UI (Method Switcher & Slider) |
| **4:30 - 5:45** | 5. Explainability & Snippets | **Member 3** | Streamlit UI (Cards, Matched terms, Holdings) |
| **5:45 - 7:00** | 6. Empirical Evaluation & Error Analysis | **Member 4** | Streamlit Evaluation Tab (`results/plots/`) |
| **7:00 - 7:45** | 7. Real Limitation & Roadmap | **Member 4** | Live Query with Ambiguity & Conclusion |

---

## 🎙️ Detailed Spoken Script by Section

### 1. Problem and Track T6 Relevance (0:00 - 1:00) --- [Speaker: Member 1]
> *"Hello everyone! We are Team LegalLens, and today we are presenting our Information Retrieval system for Track T6: Vertical Search for Law, Finance, or Science.*
> 
> *Searching through legal case law is fundamentally challenging. Appellate court judgments in India run tens of pages with archaic vocabulary. If a student or lawyer searches for 'tenant eviction without notice', simple keyword matching fails because judgments often use formal statutory phrases like 'summary ejectment proceedings under Section 21 of the Rent Control Act'. On the other hand, black-box AI chatbots frequently hallucinate case citations.
> 
> *Our goal with LegalLens was to build a real, transparent Information Retrieval system that combines lexical BM25 and dense semantic similarity, providing verifiable citations, operative judicial passages, and explainable score breakdowns."*

---

### 2. Dataset & Lexical Preprocessing (1:00 - 2:00) --- [Speaker: Member 1]
*(Show VS Code displaying `data/processed/documents.json` and `src/indexing.py`)*
> *"Here is our document collection. We used the IndicLegalQA benchmark, aggregating 10,000 Q&A pairs into 1,260 canonical Indian Supreme Court judgment documents.
> 
> In `src/preprocessing.py`, our pipeline performs case-folding, stopword filtering, and Porter stemming while strictly preserving legal statutory numbers like Section 8 or Act 1956.
> 
> In `src/indexing.py`, we implemented a classical inverted index with positional postings: mapping terms to document IDs, term frequencies, and exact token offsets. This enables us to compute Robertson-Spärck Jones IDF, track document lengths, and support exact phrase matching like 'Rent Control Act'."*

---

### 3. Lexical Retrieval: TF-IDF vs. BM25 (2:00 - 3:15) --- [Speaker: Member 2]
*(Switch to Streamlit app UI running at `http://localhost:8501`)*
> *"Now let's see our retrieval models in action on our Streamlit web application.
> 
> First, we select our baseline: the TF-IDF Vector Space Model. Let's issue the query: 'Can a tenant be evicted without proper notice under rent control?'.
> 
> As you can see, TF-IDF retrieves relevant cases using sublinear log-term weighting and cosine similarity. However, notice that TF-IDF does not penalize verbose documents or saturate term frequencies.
> 
> Now, we switch to our primary classical IR model: Okapi BM25 (`src/bm25_retriever.py`). We implemented BM25 using Robertson-Spärck Jones IDF, length normalization penalty with $b = 0.75$, and $k_1 = 1.5$. Notice how the ranking sharpens: 'Atma Ram Properties Pvt. Ltd. vs. The Oriental Insurance Co. Ltd.' jumps to the top with a score of 18.15. In addition, our top-K candidate selection uses a min-heap, achieving $O(N \log K)$ efficiency."*

---

### 4. Dense Semantic Search & Hybrid Ranking (3:15 - 4:30) --- [Speaker: Member 3]
*(In Streamlit, switch to 'Dense Semantic (MiniLM)' and then 'Hybrid')*
> *"While BM25 is strong when queries share exact terms, consider when a user paraphrases a legal concept with different words.
> 
> In `src/semantic_retriever.py`, we precomputed 384-dimensional dense vectors for all 1,260 judgments using Sentence-BERT (`all-MiniLM-L6-v2`). Caching these vectors offline satisfies Non-Functional Requirement 2, keeping query latency under 20 milliseconds.
> 
> Now, let's select our flagship model: Hybrid Retrieval (`src/hybrid_ranker.py`).
> Notice the slider for $\alpha$: it controls the convex combination between normalized BM25 and normalized semantic cosine similarity.
> Because BM25 scores are unbounded and semantic scores lie in $[-1, 1]$, we apply per-query Min-Max normalization before fusion. With $\alpha = 0.75$, we retain the precise discrimination of statutory terms while semantic embeddings pull up relevant cases that use conceptual synonyms."*

---

### 5. Explainability & Passage Snippet Extraction (4:30 - 5:45) --- [Speaker: Member 3]
*(Point to the result card metrics and yellow highlighted snippet)*
> *"A key requirement of Track T6 is explainability. Look at each retrieved result card:
> 
> 1. In the score pill breakdown, the user sees the Final Hybrid Score (0.8747), along with the exact BM25 component (1.000) and the Semantic component (0.7493).
> 2. The system displays the specific matched keywords from the query.
> 3. Most importantly, look at the highlighted passage: instead of forcing the user to read a multi-page judgment, our passage retrieval module (`src/passage_retrieval.py`) scans paragraphs using query-term density and extracts the operative judicial holding answering the exact question."*

---

### 6. Quantitative Evaluation & Error Analysis (5:45 - 7:00) --- [Speaker: Member 4]
*(Click on Tab 2: 'Quantitative IR Evaluation')*
> *"Let's examine our quantitative evaluation. We evaluated our system on a held-out test split of 2,000 queries from IndicLegalQA across 1,260 cases. None of these numbers are fabricated; they are generated live by `src/evaluation.py`.
> 
> Here is our comparison table:
> - TF-IDF Baseline: P@5 = 0.1625, Recall@10 = 0.8495, MRR = 0.7516, nDCG@10 = 0.7752.
> - BM25: P@5 = 0.1647, Recall@10 = 0.8565, MRR = 0.7712, nDCG@10 = 0.7917. BM25 improves MRR by +2.6% over the baseline.
> - On the right, our $\alpha$ tuning curve on 1,000 validation queries demonstrates that setting $\alpha \in [0.70, 0.80]$ maximizes MRR up to 0.7914 and nDCG@10 to 0.8125.
> 
> In our Error Analysis:
> - Dense semantic search outperformed BM25 on 19 queries involving high vocabulary mismatch.
> - BM25 outperformed semantic search on 412 queries containing specific statute sections (e.g. Section 302 IPC) and named case titles, where exact lexical discrimination is essential."*

---

### 7. Real Limitation & Course Roadmap (7:00 - 7:45) --- [Speaker: Member 4]
*(Demonstrate a failure case in search, e.g. 'What was held regarding limitation period?')*
> *"To address the assignment requirement of demonstrating at least one genuine limitation: when queries are overly broad or ambiguous---such as 'What was held regarding limitation period?'---the system retrieves multiple cases discussing the Limitation Act, but cannot infer user intent without multi-turn clarification.
> 
> For our course project continuation, we plan to:
> 1. Implement Tiered Indexing (Champion lists) to scale retrieval past 100,000 cases.
> 2. Integrate a Citation Authority Graph (PageRank) so seminal landmark rulings receive higher static authority.
> 3. Fine-tune a domain-specific legal bi-encoder on Indian court decisions.
> 
> Thank you! All code, tests, and documentation are available in our GitHub repository."*
