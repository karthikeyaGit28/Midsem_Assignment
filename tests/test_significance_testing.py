"""Statistical edge cases and artifact provenance, using synthetic scores only."""
import hashlib
import json

import pytest

from src.significance_testing import COMPARISONS, METRICS, compute_significance, holm_adjust, save_significance


def write_fixture(tmp_path, comparison_values, baseline=.5):
    records = []
    for index, value in enumerate(comparison_values):
        metrics = {method: {metric: value for metric in METRICS} for method in COMPARISONS}
        metrics['BM25'] = {metric: baseline for metric in METRICS}
        records.append(dict(query_id=f'q{index}', metrics=metrics))
    path = tmp_path / 'queries.json'
    path.write_text(json.dumps(records), encoding='utf-8')
    return path


def test_all_ties_are_valid_and_provenance_matches(tmp_path):
    path = write_fixture(tmp_path, [.5] * 4)
    result = compute_significance(path)
    assert result['metadata']['query_count'] == 4
    assert result['metadata']['per_query_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(r['wilcoxon_p'] == r['holm_p'] == 1 for r in result['results'])
    assert all(r['ties'] == 4 for r in result['results'])
    assert all('no statistically significant' in s['finding'] for s in result['summary'].values())


def test_constant_difference_serializes_without_nan_or_infinity(tmp_path):
    path = write_fixture(tmp_path, [.75] * 8)
    result = save_significance(tmp_path / 'out', path)
    assert all(r['paired_t_stat'] is None and r['paired_t_p'] is None for r in result['results'])
    text = (tmp_path / 'out/statistical_significance.json').read_text()
    assert 'NaN' not in text and 'Infinity' not in text
    assert 'Evidence-aware BM25 has higher' in result['summary']['evidence_aware_vs_bm25_mrr']['finding']


@pytest.mark.parametrize('values', [[.5], [.5, float('nan')], [.5, 1.2]])
def test_invalid_input_is_rejected_instead_of_reported_as_nonsignificant(tmp_path, values):
    with pytest.raises(ValueError):
        compute_significance(write_fixture(tmp_path, values))


def test_holm_preserves_input_order_and_caps_at_one():
    assert holm_adjust([.04, .01, .03, 1.0]) == pytest.approx([.09, .04, .09, 1.0])


def test_saving_tests_refreshes_only_a_matching_benchmark(tmp_path):
    path = write_fixture(tmp_path, [.5] * 4)
    evaluation_path = tmp_path / 'research_evaluation.json'
    evaluation = dict(per_query_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      query_file='data\\processed\\queries_test.json', comparison='Old claim')
    evaluation_path.write_text(json.dumps(evaluation))
    result = save_significance(tmp_path, path)
    updated = json.loads(evaluation_path.read_text())
    assert updated['comparison'] == ' '.join(s['finding'] for s in result['summary'].values())
    assert updated['query_file'] == 'data/processed/queries_test.json'
    evaluation['per_query_sha256'] = 'different-benchmark'
    evaluation_path.write_text(json.dumps(evaluation))
    save_significance(tmp_path, path)
    assert json.loads(evaluation_path.read_text()) == evaluation
