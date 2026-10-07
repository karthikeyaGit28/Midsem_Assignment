"""Persistent, explicitly human-labelled top-five study, separate from dataset qrels.

Prepare once, judge in Streamlit, export or summarize with this CLI. No qrel is
copied into a human label. Re-preparation cannot silently overwrite a study.
"""
import argparse
from collections import defaultdict, deque
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile

from src.research_engine import ROOT, MODES, ResearchEngine

STUDY_PATH = ROOT / 'results/human_evaluation/human_judgments.json'
LABELS = ('Unjudged', 'Not Relevant', 'Relevant')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        temporary = Path(stream.name)
    temporary.replace(path)


def representative_queries(queries, documents, count=15):
    """Deterministic round-robin across heuristic target categories; no relevance labels."""
    groups = defaultdict(deque)
    for query in queries:
        if 7 <= len(query['query'].split()) <= 40:
            category = documents[query['relevant_doc_id']].get('category', 'General Law')
            groups[category].append({'query_id': query['query_id'], 'query': query['query'],
                                     'selection_category': category})
    selected = []
    while len(selected) < count and any(groups.values()):
        for category in sorted(groups):
            if groups[category] and len(selected) < count:
                selected.append(groups[category].popleft())
    if len(selected) != count:
        raise ValueError('Not enough eligible representative queries.')
    return selected


def make_study(engine, queries, corpus_hash, method='Evidence-aware BM25', query_source='custom'):
    if not 10 <= len(queries) <= 20:
        raise ValueError('Select 10 to 20 study queries.')
    if method not in MODES:
        raise ValueError('Human study requires an offline lexical method.')
    if len({q['query_id'] for q in queries}) != len(queries):
        raise ValueError('Study query IDs must be unique.')
    frozen = []
    for query in queries:
        if not query['query'].strip():
            raise ValueError('Study queries must be nonempty.')
        snapshot = engine.search(query['query'], method, 5)
        results = [{**r, 'rank': rank} for rank, r in enumerate(snapshot['results'], 1)]
        frozen.append({'query_id': query['query_id'], 'query': query['query'],
                       'selection_category': query.get('selection_category', 'Custom'), 'results': results})
    config = dict(method=method, top_k=5, category=None, strict_anchors=False, alpha=0.75)
    identity = dict(corpus_sha256=corpus_hash, config=config, queries=frozen)
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(schema_version=1, study_id=study_id, created_at=datetime.now(timezone.utc).isoformat(),
                query_source=query_source, **identity, judgments={}, reviews={})


def load_study(path=STUDY_PATH, corpus_hash=None):
    study = json.loads(Path(path).read_text(encoding='utf-8'))
    if study['schema_version'] != 1 or study['config']['top_k'] != 5:
        raise ValueError('Unsupported human study format.')
    if corpus_hash and study['corpus_sha256'] != corpus_hash:
        raise ValueError('Human study belongs to a different corpus. Prepare a new study in a new directory.')
    return study


def summary(study):
    rows = []
    for query in study['queries']:
        labels = study['judgments'].get(query['query_id'], {})
        values = [labels.get(r['doc_id'], {}).get('label') for r in query['results']]
        judged = sum(v in LABELS[1:] for v in values)
        relevant = values.count('Relevant')
        complete = judged == len(values) and (bool(values) or query['query_id'] in study.get('reviews', {}))
        rows.append(dict(query_id=query['query_id'], query=query['query'], returned=len(values),
                         judged=judged, relevant=relevant, non_relevant=values.count('Not Relevant'),
                         complete=complete, P_at_5=relevant / 5 if complete else None))
    complete = bool(rows) and all(r['complete'] for r in rows)
    return dict(evaluation_type='B. Human-judged evaluation', study_id=study['study_id'],
                status='Human evaluation complete' if complete else 'Human evaluation pending',
                query_count=len(rows), completed_queries=sum(r['complete'] for r in rows),
                required_pairs=sum(r['returned'] for r in rows), judged_pairs=sum(r['judged'] for r in rows),
                relevant_pairs=sum(r['relevant'] for r in rows), non_relevant_pairs=sum(r['non_relevant'] for r in rows),
                mean_P_at_5=sum(r['P_at_5'] for r in rows) / len(rows) if complete else None,
                denominator=5, per_query=rows,
                protocol='Judge whether the case-linked answer passages address the information need. '
                         'Review all available source passages when the excerpt is insufficient. '
                         'Every returned result needs an explicit human label; missing retrieval slots are nonrelevant. '
                         'No corpus-wide recall or legal correctness is inferred.')


def judgments_csv(study):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['study_id', 'query_id', 'query', 'rank', 'doc_id',
                                               'case_name', 'label', 'judge', 'updated_at'])
    writer.writeheader()
    for query in study['queries']:
        for result in query['results']:
            saved = study['judgments'].get(query['query_id'], {}).get(result['doc_id'], {})
            writer.writerow(dict(study_id=study['study_id'], query_id=query['query_id'], query=query['query'],
                                 rank=result['rank'], doc_id=result['doc_id'], case_name=result['case_name'],
                                 label=saved.get('label', ''), judge=saved.get('judge', ''), updated_at=saved.get('updated_at', '')))
    return stream.getvalue()


def export_study(study, directory):
    directory = Path(directory)
    atomic_json(directory / 'human_summary.json', summary(study))
    (directory / 'human_judgments.csv').write_text(judgments_csv(study), encoding='utf-8')
    stream = io.StringIO(newline='')
    rows = summary(study)['per_query']
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (directory / 'human_summary.csv').write_text(stream.getvalue(), encoding='utf-8')


def save_labels(path, study_id, query_id, labels, judge):
    """Reload before merging: revisiting a query preserves all other saved queries."""
    if not judge.strip():
        raise ValueError('Enter a human judge name or initials before saving.')
    study = load_study(path)
    if study['study_id'] != study_id:
        raise ValueError('Study changed since it was opened. Reload before saving.')
    query = next((q for q in study['queries'] if q['query_id'] == query_id), None)
    if query is None:
        raise ValueError('Unknown study query.')
    valid_ids = {r['doc_id'] for r in query['results']}
    if set(labels) - valid_ids or any(v not in LABELS for v in labels.values()):
        raise ValueError('Invalid document or human label.')
    saved = study['judgments'].setdefault(query_id, {})
    for doc_id, label in labels.items():
        if label == 'Unjudged':
            saved.pop(doc_id, None)
        else:
            saved[doc_id] = dict(label=label, judge=judge.strip(), updated_at=datetime.now(timezone.utc).isoformat())
    study.setdefault('reviews', {})[query_id] = dict(judge=judge.strip(), updated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(path, study)
    export_study(study, Path(path).parent)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'summary'])
    parser.add_argument('--output', type=Path, default=STUDY_PATH)
    parser.add_argument('--queries', type=Path, help='JSON list with query_id and query; 10 to 20 entries')
    parser.add_argument('--count', type=int, default=15)
    parser.add_argument('--method', choices=MODES, default='Evidence-aware BM25')
    args = parser.parse_args()
    corpus_hash = sha256(ROOT / 'data/processed/documents.json')
    if args.action == 'prepare':
        if args.output.exists():
            raise ValueError('Study already exists; use summary or choose a new --output. Existing labels are preserved.')
        engine = ResearchEngine.from_project()
        source = args.queries or ROOT / 'data/processed/queries_test.json'
        queries = json.loads(source.read_text(encoding='utf-8'))
        if not args.queries:
            queries = representative_queries(queries, engine.documents, args.count)
        study = make_study(engine, queries, corpus_hash, args.method,
                           f'{source.name}; SHA256={sha256(source)}; category round-robin' if not args.queries else f'{source.name}; SHA256={sha256(source)}')
        atomic_json(args.output, study)
    else:
        study = load_study(args.output, corpus_hash)
    export_study(study, args.output.parent)
    print(json.dumps(summary(study), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
