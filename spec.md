# LegalLens --- Explainable Hybrid Legal Search Engine

## 1. Project Overview

**Track:** T6 --- Vertical Search for Law, Finance or Science

**Project Title:** LegalLens --- Explainable Hybrid Information
Retrieval System for Indian Legal Case Search

**Goal:**\
Build a vertical search engine that retrieves the most relevant Indian
legal judgments for a user's legal query. The system will focus on
Information Retrieval (IR), ranking, evaluation, and explainability
rather than legal advice generation.

The system will allow a user to enter a natural-language legal question
and retrieve the top relevant judgments, relevant passages, relevance
scores, and an explanation of why each result was retrieved.

------------------------------------------------------------------------

## 2. Problem Statement

Legal judgments are long, terminology-heavy documents. Finding relevant
cases for a specific legal question can require searching through large
collections of documents.

Generic keyword search may fail when the user's wording differs from the
wording used in a judgment. Pure semantic search may also miss important
exact legal terms.

LegalLens addresses this problem using a **hybrid retrieval approach**
that combines:

-   Traditional lexical Information Retrieval
-   BM25 ranking
-   Semantic similarity
-   Hybrid score-based ranking
-   Relevant passage extraction
-   Explainable retrieval results

The project will demonstrate that combining lexical and semantic
retrieval can provide better legal search results than a simple lexical
baseline.

------------------------------------------------------------------------

## 3. Assignment Alignment

The project is designed around the T6 vertical-search requirement and
keeps IR as a central component.

The prototype will include:

1.  A domain-specific legal document collection.
2.  A query-processing pipeline.
3.  An inverted-index / lexical retrieval component.
4.  TF-IDF and BM25 retrieval.
5.  Semantic retrieval using sentence embeddings.
6.  Hybrid ranking.
7.  Top-K retrieval.
8.  Quantitative IR evaluation.
9.  A working demonstration.
10. A short report and recorded demonstration.

The project will not rely on a chatbot or LLM as the primary retrieval
mechanism.

------------------------------------------------------------------------

## 4. Target Users

The prototype is intended for:

-   Students learning about legal information retrieval
-   Researchers exploring legal case retrieval
-   Users who need to locate relevant judgments for a legal topic

**Important:** LegalLens is an academic IR prototype and is **not a
legal-advice system**. Search results should not be presented as
professional legal advice.

------------------------------------------------------------------------

## 5. Dataset

### Recommended Dataset

**IndicLegalQA**

The planned dataset contains Indian Supreme Court legal questions and
associated judgments and is suitable for creating a retrieval benchmark.

The project should use:

-   Legal judgments as the document collection.
-   Legal questions as search queries.
-   Known question-to-judgment relationships as relevance information
    for evaluation.

### Planned document representation

Each document should be normalized into a structure similar to:

``` text
doc_id
case_name
date
court
category
text
```

### Planned query representation

Each query should be represented as:

``` text
query_id
query
relevant_doc_id
answer
```

The exact fields available in the downloaded dataset must be inspected
before implementation. Do not invent missing metadata.

### Dataset preparation

The preprocessing pipeline should:

1.  Load the original files.
2.  Inspect the schema.
3.  Remove unusable records.
4.  Normalize text.
5.  Handle missing values.
6.  Store cleaned documents in a consistent format.
7.  Create a searchable document collection.
8.  Map evaluation queries to their relevant documents.

------------------------------------------------------------------------

## 6. Core Functional Requirements

### FR1 --- Query Input

The system shall allow the user to enter a legal search query.

Example:

``` text
Can a tenant be evicted without proper notice?
```

### FR2 --- Query Preprocessing

The system shall perform appropriate preprocessing such as:

-   Lowercasing
-   Tokenization
-   Stop-word handling
-   Optional stemming or lemmatization

Preprocessing choices must be documented and tested rather than assumed
to improve results.

### FR3 --- Lexical Retrieval

The system shall implement at least one traditional lexical retrieval
approach.

The recommended progression is:

1.  TF-IDF + cosine similarity
2.  BM25

### FR4 --- Semantic Retrieval

The system shall optionally generate sentence/document embeddings using
a pretrained sentence-transformer model.

The semantic score will be based on cosine similarity.

### FR5 --- Hybrid Retrieval

The final system shall combine lexical and semantic scores.

Example:

``` text
FinalScore =
    alpha * BM25Score
    + (1 - alpha) * SemanticScore
```

The value of `alpha` should be tuned experimentally rather than chosen
only because it looks reasonable.

### FR6 --- Ranking

Documents shall be sorted by their final retrieval score.

The system shall return Top-K results, with K configurable.

### FR7 --- Result Explanation

Each result should display:

-   Case name
-   Date, when available
-   Court, when available
-   Category, when available
-   Retrieval score
-   Relevant passage/snippet
-   Optional matching terms

### FR8 --- Relevant Passage

The system should identify and display a short passage from the
retrieved judgment that is relevant to the query.

This feature should not be allowed to replace the actual retrieval
system.

------------------------------------------------------------------------

## 7. Information Retrieval Methods

### 7.1 Baseline --- TF-IDF

Represent each judgment using TF-IDF vectors.

Calculate:

``` text
CosineSimilarity(query, document)
```

Use this as the basic retrieval baseline.

------------------------------------------------------------------------

### 7.2 BM25

Implement BM25 as the primary traditional retrieval method.

BM25 should account for:

-   Term frequency
-   Inverse document frequency
-   Document length normalization

The BM25 implementation should be evaluated against the TF-IDF baseline.

------------------------------------------------------------------------

### 7.3 Semantic Retrieval

Generate vector representations of queries and documents using a
pretrained sentence-transformer model.

Calculate:

``` text
SemanticScore =
CosineSimilarity(query_embedding, document_embedding)
```

The exact model should be recorded in the report.

------------------------------------------------------------------------

### 7.4 Hybrid Ranking

Combine BM25 and semantic scores.

Before combining scores, ensure that the component scores are normalized
or otherwise made comparable.

Example:

``` text
FinalScore =
    alpha * NormalizedBM25
    + beta * SemanticScore
```

with:

``` text
alpha + beta = 1
```

Different weight combinations should be tested.

------------------------------------------------------------------------

## 8. Optional IR Extensions

These features are optional and should only be implemented after the
core pipeline works.

### 8.1 Query Expansion

Expand important legal terms with related terminology.

Example:

``` text
tenant
→ tenancy / lessee

eviction
→ ejectment / removal
```

Query expansion must be evaluated rather than assumed to improve
performance.

### 8.2 Metadata Boosting

If reliable metadata is available, experiment with a small ranking
adjustment using:

-   Court
-   Case category
-   Date

The metadata contribution must remain smaller than the core IR
contribution.

### 8.3 Result Diversity

If multiple retrieved documents are nearly duplicates, optionally
investigate a diversity-aware reranking strategy.

------------------------------------------------------------------------

## 9. Evaluation

Evaluation is a core part of the project.

### Required comparison

At minimum, compare:

1.  TF-IDF
2.  BM25
3.  Semantic retrieval
4.  Hybrid BM25 + semantic retrieval

### Recommended metrics

-   Precision@5
-   Precision@10
-   Recall@5
-   Recall@10
-   MRR
-   nDCG@10

Use the relevance relationships available in the dataset.

### Example result table

  Method       P@5   R@5   P@10   R@10   MRR   nDCG@10
  ---------- ----- ----- ------ ------ ----- ---------
  TF-IDF       ---   ---    ---    ---   ---       ---
  BM25         ---   ---    ---    ---   ---       ---
  Semantic     ---   ---    ---    ---   ---       ---
  Hybrid       ---   ---    ---    ---   ---       ---

**Do not fabricate evaluation numbers.** All values in the final report
must come from experiments.

### Ablation study

If time permits, perform:

-   BM25 only
-   Semantic only
-   BM25 + Semantic
-   BM25 + Semantic + optional extension

This will show which component actually contributes to performance.

------------------------------------------------------------------------

## 10. System Architecture

``` text
                    USER
                      |
                      v
              Legal Search Query
                      |
                      v
             Query Preprocessing
                      |
             +--------+--------+
             |                 |
             v                 v
          BM25 Index      Query Embedding
             |                 |
             v                 v
       Lexical Scores    Semantic Scores
             |                 |
             +--------+--------+
                      |
                      v
               Score Normalization
                      |
                      v
                Hybrid Ranking
                      |
                      v
                  Top-K Cases
                      |
              +-------+-------+
              |               |
              v               v
       Relevant Passage   Explanation
              |               |
              +-------+-------+
                      |
                      v
                 Search Results
```

------------------------------------------------------------------------

## 11. Technology Stack

### Programming

-   Python

### Data Processing

-   pandas
-   NumPy

### Traditional IR

-   scikit-learn
-   rank-bm25 or an equivalent BM25 implementation

### Semantic Retrieval

-   sentence-transformers
-   PyTorch, if required by the selected model

### Interface

-   Streamlit

### Evaluation

-   scikit-learn metrics where applicable
-   Custom implementations for IR metrics if required
-   NumPy/Python for ranking calculations

### Visualization

-   Matplotlib

------------------------------------------------------------------------

## 12. Suggested Repository Structure

``` text
LegalLens/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── indexing.py
│   ├── tfidf_retriever.py
│   ├── bm25_retriever.py
│   ├── semantic_retriever.py
│   ├── hybrid_ranker.py
│   ├── passage_retrieval.py
│   ├── evaluation.py
│   └── utils.py
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── experiments.ipynb
│
├── results/
│   ├── metrics.csv
│   └── plots/
│
├── tests/
│
├── requirements.txt
├── README.md
├── spec.md
└── .gitignore
```

------------------------------------------------------------------------

## 13. User Interface

The UI should be intentionally simple because the assignment evaluates
the IR system rather than frontend design.

### Main screen

``` text
----------------------------------------------------
                    LEGALLENS
          Indian Legal Case Search
----------------------------------------------------

Enter your legal query:

[ Can a tenant be evicted without proper notice? ]

                    [ SEARCH ]

----------------------------------------------------

Results

1. Case Name
   Score: 0.87
   Court: Supreme Court
   Date: YYYY-MM-DD

   Relevant passage:
   "...relevant passage from the judgment..."

   Why this result?
   BM25: 0.82
   Semantic: 0.91
   Final: 0.87

----------------------------------------------------
```

------------------------------------------------------------------------

## 14. Non-Functional Requirements

### NFR1 --- Reproducibility

The project should provide:

-   requirements.txt
-   setup instructions
-   dataset preparation instructions
-   fixed random seeds where applicable
-   experiment configuration

### NFR2 --- Performance

The system should precompute document representations whenever possible.

Document embeddings should not be generated from scratch for every user
query.

### NFR3 --- Transparency

The system should clearly show that retrieval is based on measurable IR
scores.

### NFR4 --- Reliability

The application should gracefully handle:

-   Empty queries
-   Queries with no matching documents
-   Missing metadata
-   Malformed records

------------------------------------------------------------------------

## 15. Scope Control

### Must Have

-   Dataset preprocessing
-   Search query input
-   TF-IDF baseline
-   BM25 retrieval
-   Top-K ranking
-   Evaluation
-   Working prototype
-   Hybrid retrieval

### Should Have

-   Semantic retrieval
-   Relevant passage extraction
-   Explanation of ranking
-   Comparison graphs

### Nice to Have

-   Query expansion
-   Metadata boosting
-   Diversity reranking
-   Search filters

### Do Not Prioritize

-   Complex frontend
-   User authentication
-   LLM-based legal advice
-   Fine-tuning a large language model
-   Large-scale web scraping
-   Training a retrieval model from scratch

------------------------------------------------------------------------

## 16. Team Division for 4 Members

### Member 1 --- Dataset & Preprocessing

Responsibilities:

-   Dataset download
-   Schema inspection
-   Cleaning
-   Document/query preparation
-   Data documentation

### Member 2 --- IR Retrieval

Responsibilities:

-   TF-IDF
-   Inverted-index related components
-   BM25
-   Retrieval pipeline

### Member 3 --- Semantic & Hybrid Ranking

Responsibilities:

-   Sentence embeddings
-   Semantic retrieval
-   Score normalization
-   Hybrid ranking
-   Optional query expansion

### Member 4 --- Evaluation & Demo

Responsibilities:

-   Evaluation metrics
-   Baseline comparison
-   Graphs/tables
-   Streamlit interface
-   Demo integration

All members should understand the complete pipeline before the final
presentation.

------------------------------------------------------------------------

## 17. Development Plan

### Phase 1 --- Setup

-   Download dataset
-   Inspect files
-   Confirm document/query relationships
-   Create repository

### Phase 2 --- Baseline

-   Implement preprocessing
-   Build TF-IDF retrieval
-   Test sample queries

### Phase 3 --- Strong IR

-   Implement BM25
-   Compare against TF-IDF

### Phase 4 --- Semantic Retrieval

-   Select pretrained sentence-transformer
-   Generate document embeddings
-   Implement semantic retrieval

### Phase 5 --- Hybrid System

-   Normalize scores
-   Combine BM25 and semantic scores
-   Tune weights using a validation split

### Phase 6 --- Evaluation

-   Run all queries
-   Calculate metrics
-   Create comparison tables and graphs

### Phase 7 --- Demo

-   Build simple Streamlit UI
-   Add snippets
-   Add ranking explanation
-   Test end-to-end

### Phase 8 --- Submission

Prepare:

-   GitHub repository
-   README
-   Report
-   Demo video
-   Results

------------------------------------------------------------------------

## 18. 36-Hour Priority Plan

### Hours 0--4

Dataset setup and preprocessing.

### Hours 4--8

TF-IDF baseline.

### Hours 8--12

BM25 retrieval.

### Hours 12--17

Evaluation framework.

### Hours 17--22

Semantic retrieval.

### Hours 22--26

Hybrid ranking.

### Hours 26--30

Streamlit interface and snippets.

### Hours 30--33

Final experiments and graphs.

### Hours 33--36

Report, README, testing, and demo recording.

If time becomes limited, **finish the baseline, BM25, evaluation, and
working demo before adding optional features.**

------------------------------------------------------------------------

## 19. Expected Demo Flow

The 5--8 minute demo should follow this sequence:

### 1. Problem

Explain why searching long legal judgments is difficult.

### 2. Dataset

Show the legal document collection and query/relevance information.

### 3. Baseline

Demonstrate TF-IDF retrieval.

### 4. Improvement

Show BM25 results.

### 5. Semantic Search

Show how semantic retrieval handles wording differences.

### 6. Hybrid System

Demonstrate the final ranking.

### 7. Explainability

Show the relevant passage and component scores.

### 8. Evaluation

Show the comparison table/graph.

### 9. Conclusion

Explain whether the hybrid system improved retrieval performance.

------------------------------------------------------------------------

## 20. Report Structure

The report should be concise and focus on the IR contribution.

Recommended sections:

1.  Introduction
2.  Problem Definition
3.  Dataset
4.  System Architecture
5.  IR Methods
6.  Baselines
7.  Hybrid Ranking Method
8.  Evaluation Methodology
9.  Results
10. Error Analysis
11. Limitations
12. Conclusion
13. References

Include a clear comparison between the baseline and final method.

------------------------------------------------------------------------

## 21. Error Analysis

Do not only report metrics.

Inspect failed queries and categorize errors such as:

-   Exact terminology mismatch
-   Legal synonym mismatch
-   Very long documents
-   Multiple legally related cases
-   Missing metadata
-   Semantic similarity without sufficient legal relevance

Use a few concrete examples in the report.

------------------------------------------------------------------------

## 22. Success Criteria

The project is considered successful if:

-   Users can search the legal collection.
-   The system returns ranked results.
-   BM25 performs competitively against the baseline.
-   Semantic retrieval can retrieve relevant cases with different
    wording.
-   The hybrid approach can be evaluated against the individual methods.
-   The evaluation is reproducible.
-   The demo shows a genuine working system.
-   The report clearly explains the IR contribution.

The final claim should be based on experimental results, not
assumptions.

------------------------------------------------------------------------

## 23. Risks and Mitigation

  -----------------------------------------------------------------------
  Risk                                Mitigation
  ----------------------------------- -----------------------------------
  Dataset schema differs from         Inspect actual files before coding
  expectations                        

  Long judgments slow semantic        Precompute document embeddings
  encoding                            

  Hybrid scores are on different      Normalize scores before combining
  scales                              

  Semantic search retrieves broadly   Keep BM25 as a strong lexical
  related but irrelevant cases        component

  Evaluation labels are incomplete    Clearly document the available
                                      relevance information

  UI consumes too much time           Keep Streamlit minimal

  Optional features delay core system Implement them only after the
                                      baseline and evaluation work
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 24. Final Project Definition

**LegalLens is an academic vertical-search system for Indian legal
judgments. It uses traditional and semantic Information Retrieval to
rank relevant cases for natural-language legal queries. The system
compares TF-IDF, BM25, semantic retrieval, and a hybrid ranking
approach, while providing relevant passages and ranking explanations.**

The central research question is:

> **Does combining lexical BM25 retrieval with semantic similarity
> improve the retrieval of relevant Indian legal judgments compared with
> individual retrieval approaches?**

This question should guide the implementation, experiments, report, and
final presentation.
