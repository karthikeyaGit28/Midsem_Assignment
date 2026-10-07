"""Reject stale or inconsistent evaluation, study, example, and report artifacts."""
import csv
import json
import math
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

from src.evaluate_research import query_metrics
from src.human_evaluation import ROOT, STUDY_PATH, atomic_json, load_study, summary, sha256
from src.significance_testing import compute_significance


def verify():
    result_dir = ROOT / 'results'
    meta = json.loads((result_dir / 'research_evaluation.json').read_text(encoding='utf-8'))
    queries = json.loads((ROOT / meta['query_file']).read_text(encoding='utf-8'))
    documents = json.loads((ROOT / 'data/processed/documents.json').read_text(encoding='utf-8'))
    traces = json.loads((result_dir / meta['per_query_file']).read_text(encoding='utf-8'))
    checks = {}
    def require(name, condition):
        checks[name] = 'PASS' if condition else 'FAIL'
        if not condition:
            raise ValueError(f'Verification failed: {name}')
    require('full_test_query_count', meta['query_count'] == len(queries) == len(traces) == 2000)
    require('document_count', meta['document_count'] == len(documents))
    require('corpus_hash', meta['corpus_sha256'] == sha256(ROOT / 'data/processed/documents.json'))
    require('query_hash', meta['query_sha256'] == sha256(ROOT / meta['query_file']))
    require('per_query_hash', meta['per_query_sha256'] == sha256(result_dir / meta['per_query_file']))
    require('metrics_hash', meta['metrics_sha256'] == sha256(result_dir / 'research_metrics.csv'))
    for file, digest in meta['source_sha256'].items():
        require('source_hash:' + file, digest == sha256(ROOT / file))
    require('trace_query_alignment', [(t['query_id'], t['query'], t['relevant_doc_id']) for t in traces] ==
            [(q['query_id'], q['query'], q['relevant_doc_id']) for q in queries])
    for method in meta['methods']:
        aggregate = next(r for r in meta['metrics'] if r['Method'] == method)
        for metric in aggregate.keys() - {'Method'}:
            values = [query_metrics(t['top_10'][method], t['relevant_doc_id'])[metric] for t in traces]
            require(f'aggregate:{method}:{metric}', math.isclose(aggregate[metric], sum(values) / len(values), abs_tol=1e-14))
        require('trace_metrics:' + method, all(t['metrics'][method] == query_metrics(t['top_10'][method], t['relevant_doc_id']) for t in traces))
    with (result_dir / 'research_metrics.csv').open(encoding='utf-8', newline='') as stream:
        saved = list(csv.DictReader(stream))
    require('csv_json_agreement', all(a['Method'] == b['Method'] and all(float(a[k]) == b[k] for k in b if k != 'Method')
                                      for a, b in zip(saved, meta['metrics'])) and len(saved) == len(meta['metrics']))
    require('deterministic_repeat_checks', meta['deterministic_repeat_checks'] == 80)
    significance = json.loads((result_dir / 'statistical_significance.json').read_text(encoding='utf-8'))
    require('statistics_query_count', significance['metadata']['query_count'] == len(traces))
    require('statistics_input_hash', significance['metadata']['per_query_sha256'] == meta['per_query_sha256'])
    require('statistics_source_hash', significance['metadata']['source_sha256'] == sha256(ROOT / 'src/significance_testing.py'))
    recomputed = compute_significance(result_dir / meta['per_query_file'])
    require('statistics_results_current', significance['results'] == recomputed['results'])
    require('statistics_summary_current', significance['summary'] == recomputed['summary'])
    with (result_dir / 'statistical_significance.csv').open(encoding='utf-8', newline='') as stream:
        saved_statistics = list(csv.DictReader(stream))
    require('statistics_csv_json_agreement', saved_statistics ==
            [{k: str(v) if v is not None else '' for k, v in row.items()} for row in significance['results']])
    for name in ('ranking_lab_example.json', 'failure_cases.json'):
        evidence = json.loads((result_dir / name).read_text(encoding='utf-8'))
        require('example_corpus:' + name, evidence['corpus_sha256'] == meta['corpus_sha256'])
    example = json.loads((result_dir / 'ranking_lab_example.json').read_text(encoding='utf-8'))
    require('worked_example_query', any(q['query_id'] == example['query_id'] and q['query'] == example['original_query'] for q in queries))
    study = load_study(STUDY_PATH, meta['corpus_sha256'])
    human = summary(study)
    require('human_summary_current', json.loads((STUDY_PATH.parent / 'human_summary.json').read_text(encoding='utf-8')) == human)
    grouping = json.loads((result_dir / 'grouping_diagnostic.json').read_text(encoding='utf-8'))
    require('grouping_input_current', grouping['raw_sha256'] == sha256(ROOT / grouping['raw_file']))
    require('grouping_agreement', grouping['processed_grouping_agrees'] and grouping['local_case_date_groups'] == len(documents))
    tests = ET.parse(result_dir / 'test_results.xml').getroot().find('testsuite')
    require('tests_passed', int(tests.attrib['failures']) == int(tests.attrib['errors']) == 0)
    report_facts = result_dir / 'report_facts.json'
    if report_facts.exists():
        facts = json.loads(report_facts.read_text(encoding='utf-8'))
        require('report_metrics_agree', facts['metrics'] == meta['metrics'])
        require('report_human_status_current', facts['human_summary'] == human)
        require('report_input_hashes', all(sha256(ROOT / path) == digest for path, digest in facts['input_sha256'].items()))
        require('report_pdf_hash', facts['pdf_sha256'] == sha256(ROOT / 'REPORT_ENHANCED.pdf'))
    return dict(status='PASS', verified_at=datetime.now(timezone.utc).isoformat(), checks=checks,
                tests=int(tests.attrib['tests']), failures=0, errors=0,
                human_status=human['status'], judged_pairs=human['judged_pairs'], required_pairs=human['required_pairs'],
                limitations=['Ranking repeat checks cover 20 queries x 4 methods, not every possible query.',
                             'The publisher/local judgment-count discrepancy remains unresolved without source judgment IDs.',
                             'Paired tests describe this answer-summary benchmark, not independent full-judgment retrieval.',
                             'This verifies artifacts, not legal correctness or the quality of human judgments.'])


def main():
    report = verify()
    atomic_json(ROOT / 'results/submission_verification.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
