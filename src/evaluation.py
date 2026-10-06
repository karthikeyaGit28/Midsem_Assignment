"""
LegalLens Comprehensive Evaluation Suite
Executes empirical Information Retrieval benchmarking strictly adhering to spec.md Section 9 & 21.
Computes:
- Precision@5, Precision@10
- Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@10)
Performs alpha weight tuning, ablation studies, error analysis, and publication-ready plotting.
"""

import os
import sys
import math
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tfidf_retriever import TFIDFRetriever
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.hybrid_ranker import HybridRanker
from src.utils import load_json, save_json, min_max_normalize


def calculate_metrics_for_query(ranked_doc_ids: List[str], ground_truth_doc_id: str) -> Dict[str, float]:
    """Calculate P@5, P@10, R@5, R@10, MRR, and nDCG@10 for a single query."""
    k_max = len(ranked_doc_ids)
    
    # Position of ground truth in ranking (1-indexed)
    rank = None
    if ground_truth_doc_id in ranked_doc_ids:
        rank = ranked_doc_ids.index(ground_truth_doc_id) + 1

    # Precision@5 and Recall@5
    hit_5 = 1.0 if (rank is not None and rank <= 5) else 0.0
    p5 = hit_5 / 5.0
    r5 = hit_5  # single relevant document per query

    # Precision@10 and Recall@10
    hit_10 = 1.0 if (rank is not None and rank <= 10) else 0.0
    p10 = hit_10 / 10.0
    r10 = hit_10

    # MRR (Mean Reciprocal Rank)
    mrr = (1.0 / rank) if rank is not None else 0.0

    # nDCG@10
    if rank is not None and rank <= 10:
        ndcg10 = 1.0 / math.log2(rank + 1.0)
    else:
        ndcg10 = 0.0

    return {
        "P@5": p5,
        "P@10": p10,
        "R@5": r5,
        "R@10": r10,
        "MRR": mrr,
        "nDCG@10": ndcg10
    }


def evaluate_ranking_predictions(
    predictions: List[List[str]],
    ground_truths: List[str]
) -> Dict[str, float]:
    """Aggregate metrics over all queries."""
    assert len(predictions) == len(ground_truths)
    all_metrics = [
        calculate_metrics_for_query(pred, gt)
        for pred, gt in zip(predictions, ground_truths)
    ]
    df = pd.DataFrame(all_metrics)
    return {col: float(df[col].mean()) for col in df.columns}


def run_alpha_tuning(
    hybrid_ranker: HybridRanker,
    val_queries: List[Dict[str, Any]],
    alpha_steps: List[float] = None
) -> pd.DataFrame:
    """Tune alpha weight on validation split over [0.0, 1.0]."""
    if alpha_steps is None:
        alpha_steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    q_texts = [q["query"] for q in val_queries]
    gts = [q["relevant_doc_id"] for q in val_queries]

    print(f"\n--- Tuning Hybrid Alpha on {len(val_queries)} Validation Queries ---")
    records = []

    # Precompute semantic and bm25 scores for all queries for fast tuning
    print("Precomputing representations for validation split...")
    q_embs = hybrid_ranker.semantic.encode_queries_batch(q_texts, batch_size=128)
    all_sem_scores = np.dot(q_embs, hybrid_ranker.semantic.doc_embeddings.T)

    all_bm25_scores = np.zeros((len(q_texts), len(hybrid_ranker.doc_ids)), dtype=np.float32)
    for i, q in enumerate(q_texts):
        all_bm25_scores[i] = hybrid_ranker.bm25.get_all_scores(q)

    # Normalize per query
    norm_bm25 = np.zeros_like(all_bm25_scores)
    norm_sem = np.zeros_like(all_sem_scores)
    for i in range(len(q_texts)):
        norm_bm25[i] = min_max_normalize(all_bm25_scores[i])
        norm_sem[i] = min_max_normalize(all_sem_scores[i])

    for a in alpha_steps:
        t0 = time.time()
        final_scores = (a * norm_bm25) + ((1.0 - a) * norm_sem)
        # Top-10
        top10_indices = np.argsort(final_scores, axis=1)[:, ::-1][:, :10]
        preds = [[hybrid_ranker.doc_ids[idx] for idx in row] for row in top10_indices]
        metrics = evaluate_ranking_predictions(preds, gts)
        metrics["alpha"] = a
        records.append(metrics)
        print(f"Alpha {a:.1f}: MRR={metrics['MRR']:.4f}, nDCG@10={metrics['nDCG@10']:.4f}, R@10={metrics['R@10']:.4f} ({time.time()-t0:.2f}s)")

    df_tuning = pd.DataFrame(records)
    best_row = df_tuning.loc[df_tuning["nDCG@10"].idxmax()]
    print(f"Optimal Alpha found: {best_row['alpha']:.1f} (nDCG@10 = {best_row['nDCG@10']:.4f})")
    return df_tuning


def run_benchmark_evaluation(
    tfidf: TFIDFRetriever,
    bm25: BM25Retriever,
    semantic: SemanticRetriever,
    hybrid: HybridRanker,
    test_queries: List[Dict[str, Any]],
    best_alpha: float = 0.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full quantitative benchmark comparing TF-IDF, BM25, Semantic, and Hybrid
    on held-out test queries (spec.md Section 9).
    """
    q_texts = [q["query"] for q in test_queries]
    gts = [q["relevant_doc_id"] for q in test_queries]
    N = len(test_queries)

    print(f"\n=======================================================")
    print(f"Running Full IR Evaluation on {N} Held-Out Test Queries")
    print(f"=======================================================")

    results = {}
    detailed_rankings = {}

    # 1. TF-IDF Baseline
    print("Evaluating 1/4: TF-IDF Baseline...")
    t0 = time.time()
    q_tfidf = tfidf.vectorizer.transform(q_texts)
    # Cosine similarity matrix: (N, D)
    tfidf_sims = q_tfidf.dot(tfidf.doc_matrix.T).toarray()
    top10_tfidf_idx = np.argsort(tfidf_sims, axis=1)[:, ::-1][:, :10]
    preds_tfidf = [[tfidf.doc_ids[idx] for idx in row] for row in top10_tfidf_idx]
    results["TF-IDF"] = evaluate_ranking_predictions(preds_tfidf, gts)
    detailed_rankings["TF-IDF"] = preds_tfidf
    print(f"TF-IDF evaluated in {time.time()-t0:.2f}s: MRR = {results['TF-IDF']['MRR']:.4f}")

    # 2. BM25
    print("Evaluating 2/4: BM25 Retriever...")
    t0 = time.time()
    all_bm25_scores = np.zeros((N, len(bm25.doc_ids)), dtype=np.float32)
    for i, q in enumerate(q_texts):
        all_bm25_scores[i] = bm25.get_all_scores(q)
    top10_bm25_idx = np.argsort(all_bm25_scores, axis=1)[:, ::-1][:, :10]
    preds_bm25 = [[bm25.doc_ids[idx] for idx in row] for row in top10_bm25_idx]
    results["BM25"] = evaluate_ranking_predictions(preds_bm25, gts)
    detailed_rankings["BM25"] = preds_bm25
    print(f"BM25 evaluated in {time.time()-t0:.2f}s: MRR = {results['BM25']['MRR']:.4f}")

    # 3. Dense Semantic
    print("Evaluating 3/4: Dense Semantic Retriever (MiniLM)...")
    t0 = time.time()
    q_embs = semantic.encode_queries_batch(q_texts, batch_size=128)
    all_sem_scores = np.dot(q_embs, semantic.doc_embeddings.T)
    top10_sem_idx = np.argsort(all_sem_scores, axis=1)[:, ::-1][:, :10]
    preds_sem = [[semantic.doc_ids[idx] for idx in row] for row in top10_sem_idx]
    results["Semantic"] = evaluate_ranking_predictions(preds_sem, gts)
    detailed_rankings["Semantic"] = preds_sem
    print(f"Semantic evaluated in {time.time()-t0:.2f}s: MRR = {results['Semantic']['MRR']:.4f}")

    # 4. Hybrid (Optimal Alpha)
    print(f"Evaluating 4/4: Hybrid (BM25 + Semantic, alpha={best_alpha:.2f})...")
    t0 = time.time()
    norm_bm25 = np.zeros_like(all_bm25_scores)
    norm_sem = np.zeros_like(all_sem_scores)
    for i in range(N):
        norm_bm25[i] = min_max_normalize(all_bm25_scores[i])
        norm_sem[i] = min_max_normalize(all_sem_scores[i])

    final_scores = (best_alpha * norm_bm25) + ((1.0 - best_alpha) * norm_sem)
    top10_hyb_idx = np.argsort(final_scores, axis=1)[:, ::-1][:, :10]
    preds_hyb = [[hybrid.doc_ids[idx] for idx in row] for row in top10_hyb_idx]
    results["Hybrid"] = evaluate_ranking_predictions(preds_hyb, gts)
    detailed_rankings["Hybrid"] = preds_hyb
    print(f"Hybrid evaluated in {time.time()-t0:.2f}s: MRR = {results['Hybrid']['MRR']:.4f}")

    # Build Comparison DataFrame
    metrics_cols = ["P@5", "R@5", "P@10", "R@10", "MRR", "nDCG@10"]
    summary_data = []
    for model_name, m_dict in results.items():
        row = {"Method": model_name}
        for col in metrics_cols:
            row[col] = round(m_dict[col], 4)
        summary_data.append(row)

    df_metrics = pd.DataFrame(summary_data)
    print("\n--- Final IR Comparison Table ---")
    print(df_metrics.to_string(index=False))

    return df_metrics, {
        "detailed_rankings": detailed_rankings,
        "ground_truths": gts,
        "queries": test_queries
    }


def perform_error_analysis(
    eval_cache: Dict[str, Any],
    bm25: BM25Retriever,
    output_path: str = "results/error_analysis.json"
) -> Dict[str, Any]:
    """
    Performs in-depth Error Analysis (spec.md Section 21) categorizing query behaviors:
    1. Semantic Wins (BM25 misses, Semantic retrieves) -> Wording / Synonym mismatch
    2. BM25 Wins (Semantic misses, BM25 retrieves) -> Exact statutory / citation specificity
    3. Both Fail -> Vocabulary gap / ambiguous query
    4. Both Succeed -> Clear, high-information queries
    """
    detailed = eval_cache["detailed_rankings"]
    gts = eval_cache["ground_truths"]
    queries = eval_cache["queries"]

    preds_bm25 = detailed["BM25"]
    preds_sem = detailed["Semantic"]
    preds_hyb = detailed["Hybrid"]

    categories = {
        "semantic_wins": [],
        "bm25_wins": [],
        "hybrid_wins_over_both": [],
        "both_failed": []
    }

    for i, q_dict in enumerate(queries):
        gt = gts[i]
        q_text = q_dict["query"]
        case_name = q_dict.get("case_name", "")

        bm25_hit = gt in preds_bm25[i][:10]
        sem_hit = gt in preds_sem[i][:10]
        hyb_hit = gt in preds_hyb[i][:10]

        bm25_rank = preds_bm25[i].index(gt) + 1 if bm25_hit else -1
        sem_rank = preds_sem[i].index(gt) + 1 if sem_hit else -1
        hyb_rank = preds_hyb[i].index(gt) + 1 if hyb_hit else -1

        item = {
            "query": q_text,
            "target_case": case_name,
            "target_doc_id": gt,
            "bm25_rank": bm25_rank,
            "semantic_rank": sem_rank,
            "hybrid_rank": hyb_rank,
            "ground_truth_answer": q_dict.get("ground_truth_answer", "")[:200]
        }

        if not bm25_hit and sem_hit:
            categories["semantic_wins"].append(item)
        elif bm25_hit and not sem_hit:
            categories["bm25_wins"].append(item)
        elif not bm25_hit and not sem_hit and hyb_hit:
            categories["hybrid_wins_over_both"].append(item)
        elif not bm25_hit and not sem_hit and not hyb_hit:
            categories["both_failed"].append(item)

    summary = {
        "total_analyzed": len(queries),
        "semantic_wins_count": len(categories["semantic_wins"]),
        "bm25_wins_count": len(categories["bm25_wins"]),
        "hybrid_rescued_count": len(categories["hybrid_wins_over_both"]),
        "both_failed_count": len(categories["both_failed"]),
        "sample_semantic_wins": categories["semantic_wins"][:5],
        "sample_bm25_wins": categories["bm25_wins"][:5],
        "sample_hybrid_rescued": categories["hybrid_wins_over_both"][:5],
        "sample_failures": categories["both_failed"][:5]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nError Analysis saved to {output_path}")
    print(f"  - Semantic Won over BM25 on: {summary['semantic_wins_count']} queries")
    print(f"  - BM25 Won over Semantic on: {summary['bm25_wins_count']} queries")
    print(f"  - Hybrid Rescued (where both missed Top-10 individually): {summary['hybrid_rescued_count']} queries")
    print(f"  - Total Mutual Failures: {summary['both_failed_count']} queries")

    return summary


def plot_evaluation_charts(df_metrics: pd.DataFrame, df_tuning: pd.DataFrame, output_dir: str = "results/plots"):
    """Generate high-quality comparison graphs for report and presentation."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Bar Chart: Model Comparison across Core Metrics
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    methods = df_metrics["Method"].tolist()
    metrics_to_plot = ["P@5", "R@5", "P@10", "R@10", "MRR", "nDCG@10"]
    x = np.arange(len(metrics_to_plot))
    width = 0.18

    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]

    for i, method in enumerate(methods):
        row = df_metrics[df_metrics["Method"] == method].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        ax.bar(x + (i - 1.5) * width, vals, width, label=method, color=colors[i % len(colors)], alpha=0.9, edgecolor="black", linewidth=0.7)

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("LegalLens Information Retrieval Benchmark Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    plt.tight_layout()

    comp_plot_path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(comp_plot_path)
    plt.close(fig)
    print(f"Saved comparison chart to {comp_plot_path}")

    # 2. Line Chart: Alpha Tuning Curve
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(df_tuning["alpha"], df_tuning["nDCG@10"], marker="o", linewidth=2.5, color="#9013FE", label="nDCG@10")
    ax.plot(df_tuning["alpha"], df_tuning["MRR"], marker="s", linewidth=2.5, color="#4A90E2", label="MRR")
    ax.plot(df_tuning["alpha"], df_tuning["R@10"], marker="^", linewidth=2.0, color="#50E3C2", linestyle="--", label="Recall@10")

    best_alpha = df_tuning.loc[df_tuning["nDCG@10"].idxmax()]["alpha"]
    best_ndcg = df_tuning["nDCG@10"].max()
    ax.axvline(x=best_alpha, color="red", linestyle=":", label=f"Optimal α = {best_alpha:.1f}")
    ax.scatter([best_alpha], [best_ndcg], color="red", s=100, zorder=5)

    ax.set_xlabel("Lexical Fusion Weight (α) [BM25 vs. Semantic]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=12, fontweight="bold")
    ax.set_title("Hybrid Fusion Sensitivity: Alpha (α) Tuning Curve", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(df_tuning["alpha"])
    ax.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    tuning_plot_path = os.path.join(output_dir, "alpha_tuning_curve.png")
    fig.savefig(tuning_plot_path)
    plt.close(fig)
    print(f"Saved alpha tuning chart to {tuning_plot_path}")


def run_full_evaluation_pipeline():
    """Main orchestrator for complete evaluation."""
    print("Loading precomputed models and evaluation splits...")
    tfidf = TFIDFRetriever.load("data/processed/tfidf_vectorizer.pkl")
    bm25 = BM25Retriever.load("data/processed/bm25_retriever.pkl")
    semantic = SemanticRetriever.load("data/processed/doc_embeddings.npy", "data/processed/semantic_metadata.pkl")
    hybrid = HybridRanker(bm25, semantic, alpha=0.5)

    val_queries = load_json("data/processed/queries_val.json")
    test_queries = load_json("data/processed/queries_test.json")

    # Step 1: Tune Alpha
    df_tuning = run_alpha_tuning(hybrid, val_queries)
    best_alpha = float(df_tuning.loc[df_tuning["nDCG@10"].idxmax()]["alpha"])
    df_tuning.to_csv("results/alpha_tuning.csv", index=False)

    # Step 2: Full Benchmark on Test Set
    df_metrics, eval_cache = run_benchmark_evaluation(tfidf, bm25, semantic, hybrid, test_queries, best_alpha=best_alpha)
    df_metrics.to_csv("results/metrics.csv", index=False)

    # Step 3: Error Analysis
    perform_error_analysis(eval_cache, bm25, output_path="results/error_analysis.json")

    # Step 4: Plots
    plot_evaluation_charts(df_metrics, df_tuning)

    print("\nAll Evaluation experiments completed successfully!")


if __name__ == "__main__":
    run_full_evaluation_pipeline()
