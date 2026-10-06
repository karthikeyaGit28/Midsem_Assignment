"""
LegalLens Streamlit Interactive Web Application
Explainable Hybrid Information Retrieval System for Indian Legal Case Search.
Adheres to spec.md Section 13 (User Interface) and assignment rubric.
"""

import os
import sys
import time
import json
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.indexing
import src.bm25_retriever
import __main__
# Ensure unpickling works regardless of whether script is run directly or via streamlit
__main__.BM25Retriever = src.bm25_retriever.BM25Retriever
__main__.InvertedIndex = src.indexing.InvertedIndex

from src.hybrid_ranker import load_hybrid_system
from src.tfidf_retriever import TFIDFRetriever
from src.passage_retrieval import highlight_matched_terms
from src.extensions import expand_query, apply_metadata_boost
from src.utils import min_max_normalize


# Page configuration
st.set_page_config(
    page_title="LegalLens | Indian Legal Case Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional academic legal search appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 20px;
    }
    .case-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.1s ease-in-out;
    }
    .case-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0d47a1;
        margin-bottom: 6px;
    }
    .meta-badge {
        display: inline-block;
        background-color: #eceff1;
        color: #37474f;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 6px;
    }
    .cat-badge {
        display: inline-block;
        background-color: #e8eaf6;
        color: #283593;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .score-badge {
        display: inline-block;
        background-color: #e8f5e9;
        color: #1b5e20;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .passage-box {
        background-color: #f9fbe7;
        border-left: 4px solid #afb42b;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 10px;
        border-radius: 0 4px 4px 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .matched-tag {
        display: inline-block;
        background-color: #fff9c4;
        color: #f57f17;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing LegalLens IR Engine & Loading Inverted Indexes...")
def get_retrievers():
    """Load and cache BM25, Semantic, Hybrid, and TF-IDF models."""
    hybrid = load_hybrid_system()
    tfidf = TFIDFRetriever.load("data/processed/tfidf_vectorizer.pkl")
    return hybrid, tfidf


# Initialize backend
try:
    hybrid_ranker, tfidf_retriever = get_retrievers()
except Exception as e:
    st.error(f"Error loading indexes: {e}. Please ensure data/processed contains indexed files.")
    st.stop()


# Sidebar Navigation & Search Parameters
st.sidebar.title("⚖️ LegalLens Config")

retrieval_mode = st.sidebar.selectbox(
    "Retrieval Engine",
    [
        "Hybrid (BM25 + Semantic)",
        "BM25 (Best Matching 25)",
        "Dense Semantic (MiniLM)",
        "TF-IDF Baseline"
    ],
    index=0,
    help="Select the core Information Retrieval model used to rank judgments."
)

top_k = st.sidebar.slider("Top-K Cases", min_value=3, max_value=25, value=10, step=1)

if retrieval_mode == "Hybrid (BM25 + Semantic)":
    alpha = st.sidebar.slider(
        "Fusion Weight (α) [BM25 vs. Semantic]",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="α = 1.0 is pure BM25; α = 0.0 is pure Semantic; 0.75 represents empirical optimum."
    )
    hybrid_ranker.set_alpha(alpha)
else:
    alpha = 0.5

use_query_expansion = st.sidebar.checkbox(
    "Enable Legal Query Expansion",
    value=False,
    help="Augment query with Indian legal synonyms (e.g., tenant → tenancy / lessee)"
)

selected_category = st.sidebar.selectbox(
    "Filter by Category",
    ["All Categories", "Criminal Law", "Constitutional Law", "Civil & Property Law", "Service & Employment Law", "Commercial & Corporate Law", "Family & Personal Law", "General Law"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Track:** T6 --- Vertical Search for Law  
**Corpus:** 1,260 Supreme Court Cases  
**Benchmark:** IndicLegalQA Benchmark  
**Metrics:** P@k, R@k, MRR, nDCG@10
""")

# Main Tabs
tab_search, tab_eval, tab_arch = st.tabs([
    "🔍 Case Search & Retrieval",
    "📊 Quantitative IR Evaluation",
    "🏛️ IR Principles & Architecture"
])

# ----------------- TAB 1: CASE SEARCH -----------------
with tab_search:
    st.markdown('<div class="main-header">LegalLens --- Indian Legal Case Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Vertical Information Retrieval System powered by Hybrid BM25 Lexical + Sentence-Transformer Dense Ranking</div>', unsafe_allow_html=True)

    # Sample queries
    st.markdown("**Sample Legal Queries:**")
    sample_queries = [
        "Can a tenant be evicted without proper notice under rent control?",
        "What is the significance of Section 8 and Section 12 of the Hindu Adoptions and Maintenance Act, 1956?",
        "Can an employee claim promotion after denial by Departmental Promotion Committee?",
        "Under what circumstances can bail be cancelled under Section 439 CrPC?",
        "Validity of arbitration clause in unstamped commercial agreement"
    ]

    col1, col2, col3 = st.columns([1, 1, 1])
    q_input = None
    if col1.button("Tenancy Eviction Notice"):
        q_input = sample_queries[0]
    if col2.button("Hindu Adoptions Act"):
        q_input = sample_queries[1]
    if col3.button("Service Promotion Denial"):
        q_input = sample_queries[2]

    # Search Box
    default_text = q_input if q_input else "Can a tenant be evicted without proper notice?"
    query = st.text_input("Enter natural-language legal question:", value=default_text, key="search_query_box")

    col_btn, col_empty = st.columns([1, 5])
    search_clicked = col_btn.button("🔍 Search Cases", type="primary")

    if query.strip():
        search_query = query.strip()
        if use_query_expansion:
            search_query = expand_query(search_query)

        t_start = time.time()

        # Execute selected retrieval engine
        if retrieval_mode == "Hybrid (BM25 + Semantic)":
            raw_results = hybrid_ranker.rank_query(search_query, top_k=top_k * 2, alpha=alpha)
        elif retrieval_mode == "BM25 (Best Matching 25)":
            raw_results = hybrid_ranker.bm25.search(search_query, top_k=top_k * 2)
            # Adapt keys
            for r in raw_results:
                r["bm25_raw"] = r.get("bm25_score", 0.0)
                r["bm25_norm"] = r.get("bm25_score", 0.0)
                r["semantic_raw"] = 0.0
                r["semantic_norm"] = 0.0
        elif retrieval_mode == "Dense Semantic (MiniLM)":
            raw_results = hybrid_ranker.semantic.search(search_query, top_k=top_k * 2)
            for r in raw_results:
                r["bm25_raw"] = 0.0
                r["bm25_norm"] = 0.0
                r["semantic_raw"] = r.get("semantic_score", 0.0)
                r["semantic_norm"] = r.get("semantic_score", 0.0)
        else:  # TF-IDF
            raw_results = tfidf_retriever.search(search_query, top_k=top_k * 2)
            for r in raw_results:
                r["bm25_raw"] = r.get("final_score", 0.0)
                r["bm25_norm"] = r.get("final_score", 0.0)
                r["semantic_raw"] = 0.0
                r["semantic_norm"] = 0.0

        # Apply category filter if requested
        if selected_category != "All Categories":
            results = [r for r in raw_results if r.get("category") == selected_category][:top_k]
        else:
            results = raw_results[:top_k]

        latency_ms = (time.time() - t_start) * 1000

        st.markdown(f"**Found {len(results)} relevant judgments** in `{latency_ms:.1f} ms` using **{retrieval_mode}**")
        if use_query_expansion and search_query != query.strip():
            st.caption(f"Expanded Query: *{search_query}*")

        st.markdown("---")

        if not results:
            st.warning("No judgments matched your search query. Try broadening your keywords.")
        else:
            for idx, item in enumerate(results, start=1):
                case_name = item.get("case_name", "Unknown Case")
                court = item.get("court", "Supreme Court of India")
                date = item.get("date", "Unknown Date")
                category = item.get("category", "General Law")
                final_score = item.get("final_score", 0.0)
                bm25_norm = item.get("bm25_norm", item.get("bm25_score", 0.0))
                sem_norm = item.get("semantic_norm", item.get("semantic_score", 0.0))
                passage = item.get("passage", "")
                matched = item.get("matched_terms", [])
                full_text = item.get("full_text", "")

                with st.container():
                    st.markdown(f"""
                    <div class="case-card">
                        <div class="case-title">{idx}. {case_name}</div>
                        <div>
                            <span class="meta-badge">🏛️ {court}</span>
                            <span class="meta-badge">📅 {date}</span>
                            <span class="cat-badge">📂 {category}</span>
                            <span class="score-badge">⭐ Score: {final_score:.4f}</span>
                        </div>
                        <div class="passage-box">
                            <strong>📌 Relevant Passage / Judicial Holding:</strong><br/>
                            "{passage}"
                        </div>
                    """, unsafe_allow_html=True)

                    # Explainability Pill Breakdown (FR7)
                    c_exp1, c_exp2, c_exp3, c_exp4 = st.columns([1.2, 1.2, 1.2, 2.5])
                    c_exp1.metric("Final Score", f"{final_score:.4f}")
                    c_exp2.metric("BM25 Component", f"{bm25_norm:.4f}")
                    c_exp3.metric("Semantic Component", f"{sem_norm:.4f}")

                    if matched:
                        c_exp4.write("**Matched Keywords:**")
                        tags_html = " ".join([f"<span class='matched-tag'>{t}</span>" for t in matched])
                        c_exp4.markdown(tags_html, unsafe_allow_html=True)

                    with st.expander("📖 View Case Summary & Additional Passages"):
                        st.write(full_text if full_text else passage)
                        st.caption(f"Document ID: {item.get('doc_id')}")

                    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: EVALUATION DASHBOARD -----------------
with tab_eval:
    st.header("📊 Quantitative Information Retrieval Evaluation")
    st.markdown("""
    Evaluation was conducted on a held-out test split of **2,000 queries** from the **IndicLegalQA** benchmark 
    across 1,260 Supreme Court cases. All numbers are computed empirically via `src/evaluation.py`.
    """)

    # Load metrics from file
    metrics_file = "results/metrics.csv"
    if os.path.exists(metrics_file):
        df_metrics = pd.read_csv(metrics_file)
        st.subheader("1. Retrieval Benchmark Comparison Table")
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, subset=["P@5", "R@5", "P@10", "R@10", "MRR", "nDCG@10"], color="#c8e6c9"),
            use_container_width=True
        )

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("2. Metric Comparison Across IR Models")
            if os.path.exists("results/plots/model_comparison.png"):
                st.image("results/plots/model_comparison.png", use_container_width=True)
        with col_p2:
            st.subheader("3. Hybrid Fusion Weight (α) Tuning Curve")
            if os.path.exists("results/plots/alpha_tuning_curve.png"):
                st.image("results/plots/alpha_tuning_curve.png", use_container_width=True)

        st.markdown("---")
        st.subheader("4. Error Analysis & Case Categorization (spec.md Section 21)")
        err_file = "results/error_analysis.json"
        if os.path.exists(err_file):
            with open(err_file, "r", encoding="utf-8") as f:
                err_data = json.load(f)

            c_c1, c_c2, c_c3, c_c4 = st.columns(4)
            c_c1.metric("Total Test Queries", err_data["total_analyzed"])
            c_c2.metric("BM25 Outperformed Semantic", err_data["bm25_wins_count"])
            c_c3.metric("Semantic Outperformed BM25", err_data["semantic_wins_count"])
            c_c4.metric("Mutual Failures", err_data["both_failed_count"])

            st.markdown("""
            **Key Findings from Error Analysis:**
            - **Exact Legal Citations & Names:** BM25 achieves high MRR on queries containing explicit statute sections (e.g. *Section 12 HAMA*, *Section 302 IPC*) and case titles, where lexical frequency discrimination is decisive.
            - **Paraphrased / Conceptual Questions:** Dense semantic retrieval rescues queries with vocabulary mismatch (e.g., *'dispossession without notice'* vs. *'ejectment'*).
            - **Hybrid Fusion:** Linear combination ensures that neither exact statutory matches nor conceptual synonyms are lost.
            """)

            with st.expander("🔍 View Concrete Error Cases"):
                st.markdown("**Sample Queries where Semantic Succeeded but BM25 Missed:**")
                for item in err_data.get("sample_semantic_wins", [])[:3]:
                    st.write(f"- **Query:** *{item['query']}*")
                    st.write(f"  Target Case: `{item['target_case']}` | Semantic Rank: `{item['semantic_rank']}` | BM25 Rank: `{item['bm25_rank']}`")

                st.markdown("**Sample Queries where BM25 Succeeded but Semantic Missed:**")
                for item in err_data.get("sample_bm25_wins", [])[:3]:
                    st.write(f"- **Query:** *{item['query']}*")
                    st.write(f"  Target Case: `{item['target_case']}` | BM25 Rank: `{item['bm25_rank']}` | Semantic Rank: `{item['semantic_rank']}`")
    else:
        st.warning("Metrics file not found. Run `python -m src.evaluation` to generate benchmark results.")

# ----------------- TAB 3: ARCHITECTURE & IR PRINCIPLES -----------------
with tab_arch:
    st.header("🏛️ IR Principles & System Architecture")
    st.markdown(r"""
    ### Alignment with CSD358 IR Lecture Syllabus:
    1. **Boolean & Positional Indexing:**
       - Dictionary + Inverted Postings structure with term frequencies and positions (`src/indexing.py`).
       - Phrase query verification via positional intersection.
    2. **Vector Space Model (TF-IDF Baseline):**
       - Sublinear term-frequency log weighting $1 + \ln(\text{tf})$.
       - Cosine similarity with $L_2$ document length normalization.
    3. **Probabilistic Retrieval (Okapi BM25):**
       - Robertson-Spärck Jones IDF: $\ln((N - df + 0.5)/(df + 0.5) + 1)$.
       - Document length normalization penalty ($b = 0.75$) and term saturation ($k_1 = 1.5$).
       - Min-heap based Top-K result assembly ($O(N \log K)$).
    4. **Dense Semantic Retrieval:**
       - Precomputed sentence embeddings via `all-MiniLM-L6-v2` (384-dimensional dense vectors).
       - Fast inner product cosine scoring over precomputed document representations.
    5. **Hybrid Fusion & Score Normalization:**
       - Min-Max per-query score normalization mapping distinct score distributions to $[0.0, 1.0]$.
       - Convex combination: $S_{\text{hybrid}} = \alpha \cdot \hat{S}_{\text{BM25}} + (1 - \alpha) \cdot \hat{S}_{\text{semantic}}$.
    6. **Snippet & Passage Extraction:**
       - Paragraph-level term density scoring extracting judicial holdings.
    """)

    st.code("""
                    USER LEGAL QUERY
                           |
                           v
                  Query Preprocessing
                  (Tokenize, Stem, Stopwords)
                           |
            +--------------+--------------+
            |                             |
            v                             v
      BM25 Inverted Index         Sentence-BERT Embedding
      (Term Freq, Length Norm)    (384-dim Dense Vector)
            |                             |
            v                             v
       BM25 Scores                 Semantic Scores
            |                             |
            +--------------+--------------+
                           |
                           v
                   Min-Max Normalization
                           |
                           v
                     Hybrid Fusion
               Final = a*BM25 + (1-a)*Sem
                           |
                           v
                   Top-K Heap Selection
                           |
            +--------------+--------------+
            |                             |
            v                             v
       Passage Extraction          Ranking Explanation
            |                             |
            +--------------+--------------+
                           |
                           v
                 Search Results in UI
    """, language="text")
