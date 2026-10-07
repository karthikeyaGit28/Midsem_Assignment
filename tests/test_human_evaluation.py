"""Synthetic fixtures only: these labels never enter the real human study."""
import json
import pytest

from src.human_evaluation import (atomic_json, load_study, make_study, save_labels,
                                  summary, judgments_csv, representative_queries)
from src.research_engine import ResearchEngine


@pytest.fixture
def study(tmp_path):
    engine = ResearchEngine([dict(doc_id='x', case_name='Tenant source', date='2020',
                                  category='Property', text='tenant notice', passages=['tenant notice'])])
    queries = [dict(query_id=f'q{i}', query='tenant notice', relevant_doc_id='x') for i in range(10)]
    value = make_study(engine, queries, 'fixture_hash')
    path = tmp_path / 'human_judgments.json'
    atomic_json(path, value)
    return path, value


def test_no_labels_are_invented_and_mean_is_pending(study):
    path, value = study
    assert value['judgments'] == {}
    assert all('relevant_doc_id' not in q for q in value['queries'])
    report = summary(load_study(path))
    assert report['status'] == 'Human evaluation pending'
    assert report['mean_P_at_5'] is None
    assert report['judged_pairs'] == 0
    assert all(q['P_at_5'] is None for q in report['per_query'])


def test_labels_survive_disk_reload_and_merge_queries(study):
    path, value = study
    save_labels(path, value['study_id'], 'q0', {'x': 'Relevant'}, 'Synthetic fixture reviewer')
    saved = save_labels(path, value['study_id'], 'q1', {'x': 'Not Relevant'}, 'Synthetic fixture reviewer')
    assert saved['judgments']['q0']['x']['label'] == 'Relevant'
    report = summary(load_study(path))
    assert report['judged_pairs'] == 2
    assert report['mean_P_at_5'] is None
    assert report['per_query'][0]['P_at_5'] == .2  # fixed 5, although only one result
    assert report['per_query'][1]['P_at_5'] == 0
    assert json.loads((path.parent / 'human_summary.json').read_text())['status'] == report['status']
    assert 'Synthetic fixture reviewer' in judgments_csv(saved)


def test_complete_study_mean_and_counts(study):
    path, value = study
    for number in range(10):
        saved = save_labels(path, value['study_id'], f'q{number}',
                            {'x': 'Relevant' if number < 4 else 'Not Relevant'}, 'Synthetic fixture reviewer')
    report = summary(saved)
    assert report['status'] == 'Human evaluation complete'
    assert report['mean_P_at_5'] == pytest.approx(.08)
    assert (report['judged_pairs'], report['relevant_pairs'], report['non_relevant_pairs']) == (10, 4, 6)


def test_reset_label_returns_query_to_pending(study):
    path, value = study
    save_labels(path, value['study_id'], 'q0', {'x': 'Relevant'}, 'Synthetic fixture reviewer')
    saved = save_labels(path, value['study_id'], 'q0', {'x': 'Unjudged'}, 'Synthetic fixture reviewer')
    assert summary(saved)['per_query'][0]['P_at_5'] is None


def test_partial_top_five_has_no_precision_until_all_five_are_judged(tmp_path):
    engine = ResearchEngine([dict(doc_id=f'd{i}', case_name='Equal fixture source', date='2020',
                                  text='tenant notice', passages=['tenant notice']) for i in range(5)])
    value = make_study(engine, [dict(query_id=f'q{i}', query='tenant notice') for i in range(10)], 'fixture_hash')
    path = tmp_path / 'study.json'
    atomic_json(path, value)
    saved = save_labels(path, value['study_id'], 'q0', {'d0': 'Relevant', 'd1': 'Relevant'}, 'Synthetic fixture reviewer')
    assert summary(saved)['per_query'][0]['P_at_5'] is None
    saved = save_labels(path, value['study_id'], 'q0', {f'd{i}': 'Not Relevant' for i in range(2, 5)}, 'Synthetic fixture reviewer')
    assert summary(saved)['per_query'][0]['P_at_5'] == .4
    assert summary(saved)['mean_P_at_5'] is None


@pytest.mark.parametrize('labels,judge', [({'bad': 'Relevant'}, 'Fixture'), ({'x': 'Auto relevant'}, 'Fixture'), ({'x': 'Relevant'}, '')])
def test_invalid_labels_or_missing_human_identity_rejected(study, labels, judge):
    path, value = study
    with pytest.raises(ValueError):
        save_labels(path, value['study_id'], 'q0', labels, judge)
    assert load_study(path)['judgments'] == {}


def test_wrong_study_or_corpus_rejected(study):
    path, value = study
    with pytest.raises(ValueError):
        save_labels(path, 'different_study', 'q0', {'x': 'Relevant'}, 'Fixture')
    with pytest.raises(ValueError):
        load_study(path, 'changed_corpus')


def test_empty_result_list_needs_explicit_human_review(study):
    path, value = study
    value['queries'][0]['results'] = []
    atomic_json(path, value)
    assert summary(value)['per_query'][0]['P_at_5'] is None
    saved = save_labels(path, value['study_id'], 'q0', {}, 'Synthetic fixture reviewer')
    assert summary(saved)['per_query'][0]['P_at_5'] == 0


def test_selection_is_deterministic_and_qrels_are_not_human_labels():
    docs = {'x': {'category': 'A'}, 'y': {'category': 'B'}}
    queries = [dict(query_id=f'q{i}', query='A representative research question containing at least seven words',
                    relevant_doc_id='x' if i % 2 else 'y') for i in range(20)]
    selected = representative_queries(queries, docs, 15)
    assert selected == representative_queries(queries, docs, 15)
    assert len(selected) == 15
    assert all('relevant_doc_id' not in q for q in selected)
