"""Reproducible paired tests over the saved per-query retrieval benchmark."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PER_QUERY_PATH = ROOT / 'results/research_per_query.json'
COMPARISONS = ['Evidence-aware BM25', 'TF-IDF', 'Rank fusion (BM25 + TF-IDF)']
METRICS = ['MRR@10', 'nDCG@10', 'P@5', 'R@10']


def holm_adjust(p_values):
    """Holm step-down correction for the full family of comparisons."""
    order = np.argsort(p_values, kind='stable')
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = previous
    return adjusted


def describe_comparison(row):
    name = 'RRF' if row['comparison'].startswith('Rank fusion') else row['comparison']
    p = row['wilcoxon_p']
    if p >= .05:
        return (f"{name} shows no statistically significant MRR@10 difference from BM25 "
                f"(p={p:.2g}; {row['wins']} wins, {row['losses']} losses, {row['ties']} ties).")
    better, worse = ('BM25', name) if row['mean_diff'] < 0 else (name, 'BM25')
    return (f"{better} has higher MRR@10 than {worse} on this benchmark "
            f"(unadjusted Wilcoxon p={p:.3g}; Holm-adjusted p={row['holm_p']:.3g}).")


def compute_significance(per_query_path=PER_QUERY_PATH):
    path = Path(per_query_path)
    raw = path.read_bytes()
    per_query = json.loads(raw.decode('utf-8'))
    if not isinstance(per_query, list) or len(per_query) < 2:
        raise ValueError('Paired tests require at least two query records.')
    query_ids = [q.get('query_id') for q in per_query]
    if any(not qid for qid in query_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError('Per-query records need unique, nonempty query IDs.')
    baseline = 'BM25'
    values = {}
    for method in [baseline] + COMPARISONS:
        for metric in METRICS:
            try:
                array = np.array([q['metrics'][method][metric] for q in per_query], dtype=float)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'Missing or invalid metric: {method} / {metric}') from error
            if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
                raise ValueError(f'Metric values must be finite and between 0 and 1: {method} / {metric}')
            values[method, metric] = array

    results = []
    for comparison in COMPARISONS:
        for metric in METRICS:
            base_vals = values[baseline, metric]
            comp_vals = values[comparison, metric]
            diff = comp_vals - base_vals
            non_zero = diff[diff != 0]
            if len(non_zero):
                # Do not silently turn a failed computation into p=1.
                wilcoxon = stats.wilcoxon(non_zero, alternative='two-sided')
                w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                if np.all(diff == diff[0]):
                    # The paired t statistic is undefined for constant nonzero differences.
                    t_stat, t_p = None, None
                else:
                    paired_t = stats.ttest_rel(comp_vals, base_vals)
                    t_stat, t_p = float(paired_t.statistic), float(paired_t.pvalue)
            else:
                w_stat, w_p, t_stat, t_p = 0.0, 1.0, 0.0, 1.0
            results.append(dict(
                comparison=comparison, baseline=baseline, metric=metric,
                mean_baseline=float(np.mean(base_vals)), mean_comparison=float(np.mean(comp_vals)),
                mean_diff=float(np.mean(diff)), wins=int(np.sum(diff > 0)),
                losses=int(np.sum(diff < 0)), ties=int(np.sum(diff == 0)),
                wilcoxon_stat=w_stat, wilcoxon_p=w_p, paired_t_stat=t_stat, paired_t_p=t_p,
                significant_005=bool(w_p < .05), significant_001=bool(w_p < .01)))

    for row, adjusted in zip(results, holm_adjust([r['wilcoxon_p'] for r in results])):
        row.update(holm_p=adjusted, significant_holm_005=bool(adjusted < .05))
    summary = {}
    for comparison, key in zip(COMPARISONS, ['evidence_aware_vs_bm25_mrr', 'tfidf_vs_bm25_mrr', 'rrf_vs_bm25_mrr']):
        row = next(r for r in results if r['comparison'] == comparison and r['metric'] == 'MRR@10')
        summary[key] = {k: row[k] for k in ('mean_diff', 'wins', 'losses', 'ties', 'wilcoxon_p', 'holm_p')}
        summary[key]['finding'] = describe_comparison(row)
    metadata = dict(query_count=len(per_query), per_query_file=path.name,
                    per_query_sha256=hashlib.sha256(raw).hexdigest(), scipy_version=scipy.__version__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    alternative='two-sided', zero_handling='discard exact zero differences for Wilcoxon',
                    multiple_testing='Holm correction across all 12 Wilcoxon comparisons',
                    paired_t_note='Null statistics indicate undefined tests for constant nonzero differences.')
    return dict(results=results, summary=summary, metadata=metadata)


def save_significance(out_dir=ROOT / 'results', per_query_path=PER_QUERY_PATH):
    data = compute_significance(per_query_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'statistical_significance.json').write_text(
        json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    with (out_dir / 'statistical_significance.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data['results'][0]))
        writer.writeheader()
        writer.writerows(data['results'])
    # Keep the benchmark's displayed comparison synchronized with these tests.
    evaluation_path = out_dir / 'research_evaluation.json'
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        if evaluation['per_query_sha256'] == data['metadata']['per_query_sha256']:
            evaluation['comparison'] = ' '.join(s['finding'] for s in data['summary'].values())
            evaluation['query_file'] = evaluation['query_file'].replace('\\', '/')
            evaluation_path.write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved paired tests for {data["metadata"]["query_count"]} queries to {out_dir}')
    return data


if __name__ == '__main__':
    save_significance()
