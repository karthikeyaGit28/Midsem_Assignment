"""Reproduce lexical baselines and prototype extensions on actual case qrels."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import time
import importlib.metadata

from src.research_engine import ROOT, MODES, ResearchEngine


def query_metrics(ranked_ids, relevant_id):
    ranked_ids = ranked_ids[:10]
    rank = ranked_ids.index(relevant_id) + 1 if relevant_id in ranked_ids else None
    return {
        'P@5': int(rank is not None and rank <= 5) / 5,
        'R@5': int(rank is not None and rank <= 5),
        'P@10': int(rank is not None and rank <= 10) / 10,
        'R@10': int(rank is not None and rank <= 10),
        'MRR@10': 1 / rank if rank else 0,
        'nDCG@10': 1 / math.log2(rank + 1) if rank else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0, help='0 evaluates all supplied test queries')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    path = ROOT / 'data/processed/queries_test.json'
    queries = json.loads(path.read_text(encoding='utf-8'))
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise ValueError('No evaluation queries.')
    engine = ResearchEngine.from_project()
    methods = ['BM25', 'TF-IDF', 'Rank fusion (BM25 + TF-IDF)', 'Evidence-aware BM25']
    scores = {m: [] for m in methods}
    traces = []
    repeat_checks = 0
    start = time.perf_counter()
    for number, query in enumerate(queries, 1):
        trace = {'query_id': query['query_id'], 'query': query['query'], 'relevant_doc_id': query['relevant_doc_id'], 'ranks': {}, 'top_10': {}, 'metrics': {}}
        for method in methods:
            ids = engine.ranked_ids(query['query'], method, 10)
            metrics = query_metrics(ids, query['relevant_doc_id'])
            scores[method].append(metrics)
            trace['ranks'][method] = ids.index(query['relevant_doc_id']) + 1 if query['relevant_doc_id'] in ids else None
            trace['top_10'][method] = ids
            trace['metrics'][method] = metrics
            if number <= 20:
                if ids != engine.ranked_ids(query['query'], method, 10):
                    raise AssertionError(f'Nondeterministic ranking: {query["query_id"]}, {method}')
                repeat_checks += 1
        traces.append(trace)
        if number % 250 == 0:
            print(f'Evaluated {number}/{len(queries)} queries', flush=True)
    rows = []
    for method in methods:
        values = scores[method]
        rows.append({'Method': method, **{metric: sum(r[metric] for r in values) / len(values) for metric in values[0]}})
    bm = rows[0]['MRR@10']
    ea = rows[-1]['MRR@10']
    comparison = f'Evidence-aware BM25 MRR@10 differs from BM25 by {ea - bm:+.4f}. '
    comparison += 'This is a numerical gain only; significance has not been tested.' if ea > bm else 'The anchor bonus does not improve this benchmark; its benefit is inspectability and explicit structural matching.'
    destination = ROOT / 'results'
    destination.mkdir(exist_ok=True)
    prefix = 'research' if not args.limit else f'research_sample_{args.limit}'
    with (destination / f'{prefix}_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    traces_path = destination / f'{prefix}_per_query.json'
    traces_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding='utf-8')
    from src.preprocessing import ALL_STOP_WORDS
    config = dict(bm25_k1=engine.bm25.k1, bm25_b=engine.bm25.b, title_term_weight=1.3,
                  anchor_bonus=0.15, rrf_constant=60, top_k=10, category=None, strict_anchors=False,
                  tie_break='document ID ascending', evaluation_type='A. Dataset-qrel evaluation')
    metadata = dict(query_count=len(queries), document_count=len(engine.doc_ids), query_file=str(path.relative_to(ROOT)),
                    corpus_sha256=hashlib.sha256((ROOT / 'data/processed/documents.json').read_bytes()).hexdigest(),
                    query_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), created_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - start, comparison=comparison, methods=methods,
                    limitations=['Corpus contains answer passages associated with test queries.', 'Qrels label one case per query.',
                                 'MRR is truncated at 10.', 'Anchor bonus 0.15 and RRF constant 60 are fixed prototype settings, not tuned on this test split.'],
                    metrics=rows, sample_errors=[t for t in traces if t['ranks']['BM25'] != t['ranks']['Evidence-aware BM25']][:12])
    metadata.update(config=config, per_query_file=traces_path.name,
                    per_query_sha256=hashlib.sha256(traces_path.read_bytes()).hexdigest(),
                    metrics_sha256=hashlib.sha256((destination / f'{prefix}_metrics.csv').read_bytes()).hexdigest(),
                    deterministic_repeat_checks=repeat_checks, repeat_scope='First 20 queries x all four methods',
                    anchor_rank_changes=sum(t['ranks']['BM25'] != t['ranks']['Evidence-aware BM25'] for t in traces),
                    timing_scope='Query evaluation loop and artifact serialization; excludes index construction',
                    python_version=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn', 'nltk', 'streamlit']},
                    stopwords_sha256=hashlib.sha256('\n'.join(sorted(ALL_STOP_WORDS)).encode()).hexdigest(),
                    source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in [
                        'src/research_engine.py', 'src/bm25_retriever.py', 'src/tfidf_retriever.py',
                        'src/preprocessing.py', 'src/indexing.py', 'src/utils.py', 'src/evaluate_research.py']})
    (destination / f'{prefix}_evaluation.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(rows, indent=2))
    print(comparison)


if __name__ == '__main__':
    main()
