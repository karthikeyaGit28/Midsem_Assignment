"""Test search submission and human-label persistence through Streamlit reruns."""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.research_engine import ResearchEngine
from src import human_evaluation as human


def test_search_and_judgment_persistence(tmp_path):
    docs = [dict(doc_id='x', case_name='Tenant source', date='2020', category='Property',
                 text='A tenant received notice under section 30.', passages=['A tenant received notice under section 30.']),
            dict(doc_id='y', case_name='Bail source', date='2021', category='Criminal',
                 text='The accused requested bail under section 439.', passages=['The accused requested bail under section 439.'])]
    engine = ResearchEngine(docs)
    app = Path(__file__).resolve().parents[1] / 'app/streamlit_app.py'
    with patch.object(ResearchEngine, 'from_project', return_value=engine), patch.object(human, 'STUDY_PATH', tmp_path / 'absent.json'):
        at = AppTest.from_file(str(app), default_timeout=30).run()
        assert not at.exception
        at.text_input(key='query_input').set_value('tenant notice')
        next(b for b in at.button if b.label == 'Search sources →').click().run()
        assert not at.exception
        assert at.session_state['snapshot']['results'][0]['doc_id'] == 'x'
        at.radio[0].set_value('Relevant').run()
        assert len(at.session_state['judgments']) == 1
        at.text_input(key='query_input').set_value('bail')
        next(b for b in at.button if b.label == 'Search sources →').click().run()
        assert not at.exception
        at.text_input(key='query_input').set_value('tenant notice')
        next(b for b in at.button if b.label == 'Search sources →').click().run()
        assert at.radio[0].value == 'Relevant'
        assert len(at.session_state['judgments']) == 1
        at.text_input(key='query_input').set_value('"unfindable exact phrase"')
        next(b for b in at.button if b.label == 'Search sources →').click().run()
        assert not at.exception
        assert at.session_state['snapshot']['results'] == []
        at.text_input(key='query_input').set_value('"unclosed')
        next(b for b in at.button if b.label == 'Search sources →').click().run()
        assert not at.exception
        assert any('quotation' in w.value for w in at.warning)
        for ignored_query in ('the and', '?!'):
            at.text_input(key='query_input').set_value(ignored_query)
            next(b for b in at.button if b.label == 'Search sources →').click().run()
            assert not at.exception
            assert at.session_state['snapshot']['results'] == []
            assert any('specific legal term' in i.value for i in at.info)


def test_formal_human_labels_survive_new_app_session(tmp_path):
    docs = [dict(doc_id='x', case_name='Tenant source', date='2020', category='Property',
                 text='tenant notice', passages=['tenant notice'])]
    engine = ResearchEngine(docs)
    path = tmp_path / 'human_judgments.json'
    root = Path(__file__).resolve().parents[1]
    study = human.make_study(engine, [dict(query_id=f'q{i}', query='tenant notice') for i in range(10)],
                              human.sha256(root / 'data/processed/documents.json'))
    human.atomic_json(path, study)
    with patch.object(ResearchEngine, 'from_project', return_value=engine), patch.object(human, 'STUDY_PATH', path):
        at = AppTest.from_file(str(root / 'app/streamlit_app.py'), default_timeout=30).run()
        assert not at.exception
        at.text_input(key='study_judge').set_value('Synthetic fixture reviewer')
        key = f"formal_{study['study_id']}_q0_x"
        at.radio(key=key).set_value('Relevant')
        next(b for b in at.button if b.label == 'Save study judgments').click().run()
        assert not at.exception
        assert human.load_study(path)['judgments']['q0']['x']['label'] == 'Relevant'
        # New app instance restores labels from disk, beyond session-state retention.
        restarted = AppTest.from_file(str(root / 'app/streamlit_app.py'), default_timeout=30).run()
        assert not restarted.exception
        assert restarted.radio(key=key).value == 'Relevant'
        assert any('Human evaluation pending' in i.value for i in restarted.info)
