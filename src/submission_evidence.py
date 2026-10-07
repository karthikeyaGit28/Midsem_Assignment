"""Generate actual Ranking Lab, failure-case, and grouping evidence; never human labels."""
from collections import Counter, defaultdict
import json
import re

from src.human_evaluation import atomic_json, sha256
from src.preprocessing import clean_text
from src.research_engine import ROOT, ResearchEngine, parse_query


def grouping_diagnostic():
    raw_path = ROOT / 'data/raw/IndicLegalQA_Dataset_10K_Revised.json'
    records = json.loads(raw_path.read_text(encoding='utf-8'))
    docs = json.loads((ROOT / 'data/processed/documents.json').read_text(encoding='utf-8'))
    groups = Counter()
    name_dates = defaultdict(set)
    missing = Counter()
    valid = []
    for record in records:
        for field in ('case_name', 'judgement_date', 'question', 'answer'):
            if not clean_text(record.get(field, '')):
                missing[field] += 1
        if not clean_text(record.get('question', '')) or not clean_text(record.get('answer', '')):
            continue
        name = clean_text(record.get('case_name', 'Unknown Case'))
        date = clean_text(record.get('judgement_date', 'Unknown Date'))
        groups[name, date] += 1
        name_dates[name].add(date)
        valid.append(record)
    multi = [{'case_name': name, 'dates': sorted(dates),
              'qa_pairs_by_date': {date: groups[name, date] for date in sorted(dates)}}
             for name, dates in sorted(name_dates.items()) if len(dates) > 1]
    normalized = defaultdict(set)
    for name in name_dates:
        normalized[re.sub(r'[^a-z0-9]', '', name.lower())].add(name)
    collisions = [sorted(names) for names in normalized.values() if len(names) > 1]
    stored_groups = {(d['case_name'], d['date']): d for d in docs}
    agreement = set(stored_groups) == set(groups) and all(stored_groups[g]['num_qa_pairs'] == n for g, n in groups.items())
    extra_dates = sum(len(dates) - 1 for dates in name_dates.values())
    return dict(raw_file=str(raw_path.relative_to(ROOT)), raw_sha256=sha256(raw_path), raw_records=len(records),
                valid_qa_records=len(valid), missing_fields={f: missing[f] for f in ('case_name', 'judgement_date', 'question', 'answer')},
                duplicate_complete_records=len(valid) - len({json.dumps(r, sort_keys=True) for r in valid}),
                unique_cleaned_names=len(name_dates), local_case_date_groups=len(groups), extra_date_groups=extra_dates,
                multi_date_names=multi, punctuation_case_name_collisions=collisions,
                processed_document_count=len(docs), processed_grouping_agrees=agreement,
                upstream_reported_judgments=1256, upstream_source='https://data.mendeley.com/datasets/gf8n8cnmvc/2',
                conclusion=f'{len(name_dates)} cleaned names + {extra_dates} extra date groups = {len(groups)} local groups. '
                           'Missing fields and exact duplicate QA records are counted separately and do not explain this group count. '
                           'Same-name different-date entries may be separate decisions or metadata errors. '
                           'The supplied JSON has no persistent judgment ID or source URL; it cannot determine which local groups '
                           'correspond to the publisher\'s 1,256 judgments. Do not merge or correct dates without primary source evidence.')


def ranking_example(engine, query):
    experiments = engine.counterfactuals(query['query'], top_k=5)
    experiment = next((r for r in experiments if r['winner_changed']), experiments[0])
    original = engine.ranked_ids(query['query'], top_k=5)
    after = engine.ranked_ids(experiment['query'], top_k=len(engine.doc_ids))
    new_top = after[:5]
    def cases(ids):
        return [dict(rank=i, doc_id=d, case_name=engine.documents[d]['case_name']) for i, d in enumerate(ids, 1)]
    # Repeat the worked experiment, independently of saved benchmark files.
    assert original == engine.ranked_ids(query['query'], top_k=5)
    assert after == engine.ranked_ids(experiment['query'], top_k=len(engine.doc_ids))
    return dict(query_id=query['query_id'], original_query=query['query'], method='Evidence-aware BM25',
                category=None, strict_anchors=False, alpha=0.75, top_k=5,
                selection='Fixed test query q_05836; first winner-changing removal in the existing high-IDF lab order. Diagnostic, not an accuracy sample.',
                original_top_5=cases(original), removed_term=experiment['removed'], modified_query=experiment['query'],
                new_top_5=cases(new_top), original_winner_new_rank=after.index(original[0]) + 1 if original[0] in after else None,
                rank_scope='Full positive-score, eligible ranking, not truncated to top five',
                top_5_jaccard=len(set(original) & set(new_top)) / len(set(original) | set(new_top)),
                winner_changed=not new_top or new_top[0] != original[0], deterministic_repeat=True,
                interpretation='Ranking sensitivity is an IR diagnostic; it does not prove legal importance, legal causality, or correctness.')


def failure_examples(engine):
    examples = []
    for query, comparison, intended, reason in [
        ('"Hindu Adoptions and Maintenance Act" section 12', 'adoption maintenance section 12',
         'Find answer passages about the named Act and section using its full title.',
         'The required full surface phrase is absent from this answer-summary corpus. Exact constraints are never silently relaxed.'),
        ('"sec. 302"', '"section 302"', 'Treat sec. 302 as an abbreviation of section 302.',
         'Surface phrase matching does not expand abbreviations. sec and section are distinct literal tokens.')]:
        ids = engine.ranked_ids(query, top_k=5)
        alternate = engine.ranked_ids(comparison, top_k=5)
        examples.append(dict(query=query, intended=intended, observed=f'{len(ids)} returned results; alternate query returns {len(alternate)} in top five.',
                             reason=reason, returned_ids=ids, parsed_plan=parse_query(query).__dict__,
                             alternate_query=comparison, alternate_top_5=[dict(doc_id=d, case_name=engine.documents[d]['case_name']) for d in alternate]))
    return dict(corpus_sha256=sha256(ROOT / 'data/processed/documents.json'), method='Evidence-aware BM25',
                top_k=5, category=None, strict_anchors=False, examples=examples,
                boundary='Observed lexical/literal limitations, not expert judgments of legal correctness.')


def main():
    destination = ROOT / 'results'
    atomic_json(destination / 'grouping_diagnostic.json', grouping_diagnostic())
    engine = ResearchEngine.from_project()
    queries = json.loads((ROOT / 'data/processed/queries_test.json').read_text(encoding='utf-8'))
    query = next(q for q in queries if q['query_id'] == 'q_05836')
    example = ranking_example(engine, query)
    example.update(corpus_sha256=sha256(ROOT / 'data/processed/documents.json'),
                   query_sha256=sha256(ROOT / 'data/processed/queries_test.json'))
    atomic_json(destination / 'ranking_lab_example.json', example)
    atomic_json(destination / 'failure_cases.json', failure_examples(engine))
    lines = ['# Reproducible Ranking Lab example', '', f"Dataset query {example['query_id']}: {example['original_query']}",
             f"Method: {example['method']}. Removed term: {example['removed_term']}", f"Modified query: {example['modified_query']}", '',
             'Original top five:']
    for case in example['original_top_5']:
        lines.append(f"{case['rank']}. {case['case_name']} ({case['doc_id']})")
    lines += ['', 'New top five:']
    for case in example['new_top_5']:
        lines.append(f"{case['rank']}. {case['case_name']} ({case['doc_id']})")
    lines += ['', f"Original winner's full new rank: {example['original_winner_new_rank']}",
              f"Top-five Jaccard: {example['top_5_jaccard']:.6f}; winner changed: {example['winner_changed']}", '', example['interpretation']]
    (destination / 'ranking_lab_example.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(example, indent=2, ensure_ascii=False))
    print(json.dumps(grouping_diagnostic(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
