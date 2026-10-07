"""LegalLens research desk: real retrieval, evidence, and ranking experiments."""
from pathlib import Path
import hashlib
import html
import json
import sys
import time
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.research_engine import ResearchEngine, research_brief
from src.passage_retrieval import highlight_matched_terms
from src import human_evaluation as human

st.set_page_config(page_title='LegalLens | The Research Desk', page_icon='⚖', layout='wide')
st.markdown('<style>' + (ROOT / 'app/styles.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)

@st.cache_resource(show_spinner='Building lexical and positional indexes from the supplied corpus…')
def get_engine(corpus_version):
    return ResearchEngine.from_project()

@st.cache_data(show_spinner=False)
def run_search(query, mode, top_k, category, strict, alpha, engine_version, _engine):
    start = time.perf_counter()
    snapshot = _engine.search(query, mode, top_k, category, strict, alpha)
    snapshot['latency_ms'] = (time.perf_counter() - start) * 1000
    return snapshot

@st.cache_data(show_spinner=False)
def run_counterfactuals(query, mode, category, strict, alpha, engine_version, _engine):
    return _engine.counterfactuals(query, mode, 5, category, strict, alpha)

corpus_stat = (ROOT / 'data/processed/documents.json').stat()
engine = get_engine((corpus_stat.st_mtime_ns, corpus_stat.st_size))
if st.session_state.get('active_engine') != id(engine):
    st.session_state.pop('snapshot', None)
    st.session_state.pop('counterfactual_result', None)
    st.session_state['active_engine'] = id(engine)
st.session_state.setdefault('judgments', {})
with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-icon">⚖</div><div>
    <div class="brand-name">LegalLens</div><div class="brand-note">THE RESEARCH DESK</div></div></div>''', unsafe_allow_html=True)
    st.caption('CSD358 · Information Retrieval · Track T6')
    st.divider()
    mode = st.selectbox('Retrieval method', engine.modes)
    top_k = st.slider('Cases to retrieve', 3, 20, 5)
    categories = sorted({d.get('category', 'General Law') for d in engine.documents.values()})
    category = st.selectbox('Topic filter', ['All topics'] + categories)
    strict = st.checkbox('Require every section / article anchor', value=False)
    alpha = st.slider('BM25 weight', 0.0, 1.0, .75, .05) if mode == 'Hybrid (BM25 + MiniLM)' else .75
    with st.expander('Search syntax & tips'):
        st.markdown('**Exact phrase:** `"rent control act"`\n\n**Exclude a word:** `tenant -tax`\n\n**Section anchor:** `section 302`')
        st.caption('Quoted phrases are required. Topic and literal filters apply before selecting the top results. Press Search sources after changing controls.')
    st.divider()
    st.caption(f'{len(engine.documents):,} case-linked documents · {len(engine.bm25.index.postings):,} indexed terms')
    st.markdown('<div class="sidebar-note"><span class="status-dot"></span><b>Offline search ready</b><br>Search source passages, inspect scores, and export your evidence.</div>', unsafe_allow_html=True)
    with st.expander('Optional semantic search'):
        st.caption('Needs matching embeddings and a locally cached MiniLM model.')
        if engine.semantic is None and st.button('Connect local MiniLM index'):
            try:
                with st.spinner('Checking local dense assets…'):
                    engine.enable_semantic()
                run_search.clear()
                run_counterfactuals.clear()
                st.rerun()
            except (ImportError, OSError, ValueError) as error:
                st.warning(f'Dense search unavailable: {error}')

st.markdown('''<div class="desk-topline"><span>LEGAL RESEARCH / CSD358</span>
<span><span class="status-dot"></span>LOCAL CORPUS · OFFLINE READY</span></div>''', unsafe_allow_html=True)
st.markdown(f'''<div class="hero"><div class="hero-grid"><div>
<div class="eyebrow">A CLEARER VIEW OF LEGAL SOURCES</div>
<h1>Find the case.<br><em>Follow the evidence.</em></h1>
<p>Explore case-linked source passages, understand why they rank, and put your results to the test.</p>
<span class="hero-label">Exact phrase search</span><span class="hero-label">Ranking experiments</span>
<span class="hero-label">Traceable excerpts</span></div>
<div class="hero-aside"><div class="library-label">THE SOURCE LIBRARY</div>
<div class="library-count">{len(engine.documents):,}</div><div class="library-detail">case-linked documents<br>from IndicLegalQA</div>
<div class="library-footer">{len(engine.modes)} retrieval methods · one research desk</div></div>
</div></div>''', unsafe_allow_html=True)
search_tab, lab_tab, benchmark_tab, judgment_tab, pipeline_tab = st.tabs(
    ['Search sources', 'Ranking Lab', 'Benchmarks', 'Judge relevance', 'Inside the pipeline'])

with search_tab:
    st.markdown('''<div class="section-kicker">01 / SOURCE EXPLORER</div><h2 class="section-heading">What are you researching?</h2>
    <p class="section-description">Start with a legal question, a section number, or an exact phrase.</p>''', unsafe_allow_html=True)
    examples = {'Tenancy & notice': 'Can a tenant be evicted without proper notice?',
                'Exact source phrase': '"rent control act"',
                'Promotion disputes': 'denial of promotion seniority'}
    for col, (label, example) in zip(st.columns(3), examples.items()):
        if col.button(label, width='stretch'):
            st.session_state['query_input'] = example
            st.session_state['run_example'] = True
    with st.form('search_form'):
        question_col, submit_col = st.columns([4, 1.3], vertical_alignment='bottom')
        with question_col:
            query = st.text_input('Your research question', key='query_input', max_chars=2000,
                                  placeholder='e.g. bail cancellation section 439, or "rent control" -tax')
        with submit_col:
            submitted = st.form_submit_button('Search sources →', type='primary', width='stretch')
    run_example = st.session_state.pop('run_example', False)
    if submitted or run_example:
        if not query.strip():
            st.warning('Enter a question or select an example to search.')
        else:
            try:
                st.session_state['snapshot'] = run_search(query.strip(), mode, top_k, None if category == 'All topics' else category, strict, alpha, id(engine), engine)
            except ValueError as error:
                st.warning(str(error))
    snapshot = st.session_state.get('snapshot')
    if snapshot:
        st.caption(f"Last submitted search · {snapshot['mode']} · {snapshot['query']}")
        st.caption('Change controls and press Search sources to run another experiment.')
        rows, diag = snapshot['results'], snapshot['diagnostics']
        a, b, c, d = st.columns(4)
        a.metric('Retrieved sources', len(rows))
        b.metric('Retrieval time', f"{snapshot['latency_ms']:.0f} ms")
        c.metric('Top-case token coverage', f"{diag['coverage']:.0%}")
        d.metric('BM25 / TF-IDF overlap', f"{diag['overlap']:.0%}")
        st.caption('Coverage counts normalized query tokens; overlap is top-K Jaccard agreement. Neither is a probability of correctness. Time excludes index building; cached replays reuse the recorded time.')
        if rows:
            ex1, ex2, _ = st.columns([1, 1, 2])
            ex1.download_button('↓ Evidence brief', research_brief(snapshot), 'legallens-evidence.md', 'text/markdown')
            ex2.download_button('↓ Experiment JSON', json.dumps(snapshot, indent=2, ensure_ascii=False), 'legallens-experiment.json', 'application/json')
        else:
            if diag.get('reason') == 'no_searchable_terms':
                st.info('Add a specific legal term, such as tenant, bail, promotion, or a section number. This query contains only ignored words or punctuation.')
            else:
                st.info('No sources satisfy this query. Try fewer constraints or a broader topic. Exact phrases are never silently relaxed.')
        for rank, row in enumerate(rows, 1):
            evidence = row['evidence']
            highlighted = highlight_matched_terms(evidence['text'], snapshot['plan']['scoring_query'])
            badges = ''.join(f'<span class="pill">{html.escape(str(v))}</span>' for v in [row['date'], row['category'], f"Score {row['score']:.4f}"])
            st.markdown(f'''<div class="case"><div class="case-header"><div class="case-rank">{rank:02d}</div>
            <div class="rank">SOURCE EXCERPT / {html.escape(row['doc_id'])}</div></div>
            <h3>{html.escape(row['case_name'])}</h3>{badges}<div class="evidence">{highlighted}</div>
            <div class="source-line">IndicLegalQA answer passage {evidence['passage_number']} · words {evidence['word_start'] + 1}–{evidence['word_end']} · source excerpt</div></div>''', unsafe_allow_html=True)
            with st.expander(f'Inspect evidence & score · source {rank}'):
                st.write('**Matching tokens:** ' + ', '.join(row['matched_terms']))
                st.write('**Literal anchors found:** ' + (', '.join(row['anchor_matches']) or 'None requested or matched'))
                st.caption('Evidence-aware BM25 adds at most 0.15 for literal section/article coverage. This is a disclosed heuristic, not learned authority.')
                st.dataframe(pd.DataFrame(engine.term_contributions(snapshot['plan']['scoring_query'], row['doc_id'])), hide_index=True, width='stretch')
                st.write('**All source answer passages**')
                for number, passage in enumerate(engine.documents[row['doc_id']].get('passages', []), 1):
                    st.write(f'{number}. {passage}')
    else:
        st.markdown('''<div class="empty"><div class="empty-symbol">⌕</div>
        <h3>A question is a good place to start.</h3><p>Choose an example above or write your own. Your source excerpts will appear here.</p>
        <div class="workflow"><div class="workflow-step"><b>01 · Find a source</b><span>Search the local library with words, phrases, and topic filters.</span></div>
        <div class="workflow-step"><b>02 · Inspect the evidence</b><span>Read the original passage and see each term's score contribution.</span></div>
        <div class="workflow-step"><b>03 · Challenge the ranking</b><span>Compare methods and test a word removal in the Ranking Lab.</span></div></div></div>''', unsafe_allow_html=True)
    st.caption('Corpus: answer passages grouped by case from IndicLegalQA, not full judgments. Topic labels are inferred by keywords. Academic retrieval prototype.')

with lab_tab:
    st.subheader('Challenge the ranking')
    st.write('An explanation should be testable. Inspect the real scoring terms, compare retrieval methods, and remove a word to see which cases move.')
    snapshot = st.session_state.get('snapshot')
    if not snapshot or not snapshot['results']:
        st.info('Run a search with results to start a ranking experiment.')
    else:
        st.caption('Active experiment: ' + snapshot['query'])
        doc_id = st.selectbox('Inspect a source', [r['doc_id'] for r in snapshot['results']], format_func=lambda d: engine.documents[d]['case_name'])
        contributions = engine.term_contributions(snapshot['plan']['scoring_query'], doc_id)
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown('**What made this case match?**')
            if contributions:
                st.bar_chart(pd.DataFrame(contributions).set_index('term')['contribution'], color='#315970')
            else:
                st.info('No lexical overlap: this source was retrieved by the secondary signal.')
        with right:
            st.markdown('**Score audit**')
            st.metric('Sum of term contributions', f"{sum(r['contribution'] for r in contributions):.6f}")
            st.caption('IDF × saturated term frequency × query frequency × title-zone weight. Title weight is 1.3 when a term appears in the case name.')
            st.json(snapshot['plan'], expanded=True)
        st.divider()
        st.markdown('**Do the retrieval methods agree?**')
        rank_rows = []
        for method in engine.modes:
            ids = engine.ranked_ids(snapshot['query'], method, 5, snapshot['category'], snapshot['strict_anchors'], snapshot['alpha'])
            rank_rows.extend({'Method': method, 'Rank': rank, 'Case': engine.documents[d]['case_name']} for rank, d in enumerate(ids, 1))
        st.dataframe(pd.DataFrame(rank_rows), hide_index=True, width='stretch')
        if st.button('Run leave-one-term-out experiment', type='primary'):
            with st.spinner('Re-running retrieval for up to eight term removals…'):
                experiments = run_counterfactuals(snapshot['query'], snapshot['mode'], snapshot['category'], snapshot['strict_anchors'], snapshot['alpha'], id(engine), engine)
            st.session_state['counterfactual_result'] = (snapshot['query'], snapshot['mode'], snapshot['category'], snapshot['strict_anchors'], snapshot['alpha'], experiments)
        current = st.session_state.get('counterfactual_result')
        signature = (snapshot['query'], snapshot['mode'], snapshot['category'], snapshot['strict_anchors'], snapshot['alpha'])
        if current and current[:5] == signature:
            experiments = current[5]
            st.metric('Term removals that changed the top case', f"{sum(r['winner_changed'] for r in experiments)} / {len(experiments)}")
            for experiment in experiments:
                if experiment['winner_changed']:
                    st.write(f"Removing **{experiment['removed']}** changes the top case to **{experiment['top_case']}**.")
            if not experiments:
                st.info('This query has no removable scoring terms. Try a query with unquoted legal keywords; exact phrases and exclusions stay fixed.')
            comparison_frame = pd.DataFrame(experiments).rename(columns={
                'removed': 'Removed term', 'query': 'Rerun query', 'top_case': 'New top case',
                'original_winner_rank': 'Original winner rank', 'overlap': 'Top-5 Jaccard', 'winner_changed': 'Winner changed'})
            st.dataframe(comparison_frame, hide_index=True, width='stretch')
            st.caption('Original winner rank is within the new top 5; blank means it fell outside. Quoted phrases and -word exclusions stay fixed. Top-5 overlap is Jaccard, not accuracy.')
            st.download_button('↓ Counterfactual experiment', json.dumps(current[5], indent=2), 'legallens-counterfactuals.json', 'application/json')

    worked_path = ROOT / 'results/ranking_lab_example.json'
    if worked_path.exists():
        with st.expander('Saved reproducible worked example from the dataset'):
            worked = json.loads(worked_path.read_text(encoding='utf-8'))
            st.write(worked['original_query'])
            st.write(f"Removed: {worked['removed_term']} · Original winner's new rank: {worked['original_winner_new_rank']} · "
                     f"Top-five Jaccard: {worked['top_5_jaccard']:.3f} · Winner changed: {worked['winner_changed']}")
            st.dataframe(pd.DataFrame({'Original top five': [r['case_name'] for r in worked['original_top_5']],
                                       'New top five': [r['case_name'] for r in worked['new_top_5']]}), hide_index=True, width='stretch')
            st.caption(worked['interpretation'])
            st.download_button('↓ Worked Ranking Lab JSON', worked_path.read_bytes(), 'ranking_lab_example.json', 'application/json')

with benchmark_tab:
    st.subheader('Measured results, visible limitations')
    st.info('The corpus contains the dataset’s answer passages, including answers associated with evaluation queries. This measures case retrieval over answer summaries; it does not establish performance on unseen full judgments. Each query has one labelled case, so other relevant cases may be unlabelled.')
    fresh = ROOT / 'results/research_metrics.csv'
    if fresh.exists():
        df = pd.read_csv(fresh)
        st.markdown('**A. Dataset-qrel evaluation**')
        st.dataframe(df, hide_index=True, width='stretch')
        st.bar_chart(df.set_index('Method')[['MRR@10', 'nDCG@10']], color=['#315970', '#b49559'], stack=False, y_label='Score (0 to 1)')
        meta_path = ROOT / 'results/research_evaluation.json'
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding='utf-8'))
            st.caption(f"Queries: {metadata['query_count']:,} · Documents: {metadata['document_count']:,} · Query file: {metadata['query_file']} · Corpus SHA-256: {metadata['corpus_sha256'][:16]}")
            st.write('**Observed comparison:** ' + metadata['comparison'])
        significance_path = ROOT / 'results/statistical_significance.json'
        if significance_path.exists() and meta_path.exists():
            significance = json.loads(significance_path.read_text(encoding='utf-8'))
            provenance = significance.get('metadata', {})
            if (provenance.get('per_query_sha256') == metadata['per_query_sha256']
                    and provenance.get('query_count') == metadata['query_count']):
                with st.expander('Paired statistical tests · method comparisons'):
                    st.caption('Two-sided Wilcoxon signed-rank tests compare per-query scores against BM25. P-values below are unadjusted; all comparisons also include a Holm correction for multiple testing. These describe this benchmark only.')
                    for finding in significance['summary'].values():
                        st.write(finding['finding'])
                    st.dataframe(pd.DataFrame(significance['results'])[
                        ['comparison', 'metric', 'mean_diff', 'wins', 'losses', 'ties', 'wilcoxon_p', 'holm_p']],
                        hide_index=True, width='stretch')
                    st.download_button('↓ Statistical tests JSON', significance_path.read_bytes(), 'statistical_significance.json', 'application/json')
            else:
                st.caption('Statistical tests need to be regenerated for the current benchmark: python -m src.significance_testing')
    else:
        st.code('python -m src.evaluate_research', language='bash')
        st.caption('Run this command to compute the new benchmark. No new metric is hard-coded into the app.')
    with st.expander('Previous project benchmark (saved by the original team)'):
        old = ROOT / 'results/metrics.csv'
        if old.exists():
            st.dataframe(pd.read_csv(old), hide_index=True, width='stretch')
        st.caption('Historical artifacts are preserved. They do not verify the optional dense model in this checkout. The supplied hybrid test row equals BM25, so it does not demonstrate a test-set gain.')
    failures = ROOT / 'results/failure_cases.json'
    if failures.exists():
        with st.expander('Reproducible failure / limitation demonstrations'):
            for failure in json.loads(failures.read_text(encoding='utf-8'))['examples']:
                st.markdown('**Query:** ' + failure['query'])
                st.write('Intended behavior: ' + failure['intended'])
                st.write('Observed: ' + failure['observed'])
                st.write('Reason: ' + failure['reason'])
            st.download_button('↓ Failure-case evidence', failures.read_bytes(), 'failure_cases.json', 'application/json')

with judgment_tab:
    st.subheader('Exploratory judgments for the current search')
    st.write('Review the actual excerpts and label each case. Export your judgments for the assignment; these labels are yours, not generated ground truth.')
    snapshot = st.session_state.get('snapshot')
    if not snapshot or not snapshot['results']:
        st.info('Run a search first, then judge its sources here.')
    else:
        qid = hashlib.sha256(snapshot['query'].encode()).hexdigest()[:16]
        st.caption('Judging query: ' + snapshot['query'])
        ratings = []
        for row in snapshot['results']:
            st.markdown('**' + row['case_name'] + '**')
            st.write(row['evidence']['text'])
            key = f"judge_{qid}_{row['doc_id']}"
            saved = st.session_state['judgments'].get(key)
            initial = (2 if saved['relevant'] else 1) if saved else 0
            rating = st.radio('Relevance', ['Unjudged', 'Not relevant', 'Relevant'], index=initial, key=key, horizontal=True)
            if rating != 'Unjudged':
                st.session_state['judgments'][key] = dict(query_id=qid, query=snapshot['query'], doc_id=row['doc_id'], case_name=row['case_name'], relevant=int(rating == 'Relevant'))
                ratings.append(int(rating == 'Relevant'))
            else:
                st.session_state['judgments'].pop(key, None)
        if len(ratings) == len(snapshot['results']):
            st.metric(f'Judged Precision@{len(ratings)}', f'{sum(ratings) / len(ratings):.3f}')
            st.caption('Precision requires all displayed cases to be judged. Recall requires corpus-wide relevance judgments and is not estimated here.')
        else:
            st.caption(f"{len(ratings)} / {len(snapshot['results'])} sources judged. Complete the set to calculate precision.")
        export = list(st.session_state['judgments'].values())
        if export:
            st.download_button('↓ Download human judgments', json.dumps(export, indent=2, ensure_ascii=False), 'legallens-human-judgments.json', 'application/json')
            st.caption('Labels stay in this browser session. Download before closing the app.')

    st.divider()
    st.subheader('B. Human-judged evaluation')
    st.write('The formal study uses frozen top-five lists. Review the information need and original passages; '
             'label relevance to the question, not legal correctness. No dataset qrel is used as a human label.')
    if not human.STUDY_PATH.exists():
        st.info('Human evaluation pending. Prepare the study before judging.')
        st.code('python -m src.human_evaluation prepare')
    else:
        try:
            study = human.load_study(human.STUDY_PATH, human.sha256(ROOT / 'data/processed/documents.json'))
        except (ValueError, KeyError, OSError) as error:
            st.error(f'Cannot use this human study: {error}')
        else:
            study_summary = human.summary(study)
            st.info(study_summary['status'])
            st.progress(study_summary['completed_queries'] / study_summary['query_count'], text='Saved study progress')
            progress_cols = st.columns(3)
            progress_cols[0].metric('Queries reviewed', f"{study_summary['completed_queries']} / {study_summary['query_count']}")
            progress_cols[1].metric('Results judged', f"{study_summary['judged_pairs']} / {study_summary['required_pairs']}")
            progress_cols[2].metric('Mean Precision@5', f"{study_summary['mean_P_at_5']:.2f}" if study_summary['mean_P_at_5'] is not None else 'Pending')
            st.caption(f"{study_summary['completed_queries']} / {study_summary['query_count']} queries complete · "
                       f"{study_summary['judged_pairs']} / {study_summary['required_pairs']} result pairs judged · "
                       f"Method: {study['config']['method']}. Stored in results/human_evaluation/. Sidebar controls do not change this frozen study.")
            selected = st.selectbox('Study query', [q['query_id'] for q in study['queries']],
                                    format_func=lambda qid: next(qid + ' · ' + q['query'] for q in study['queries'] if q['query_id'] == qid),
                                    key='study_query')
            study_query = next(q for q in study['queries'] if q['query_id'] == selected)
            st.write('**Information need:** ' + study_query['query'])
            with st.form('formal_human_judging'):
                judge = st.text_input('Judge name or initials', key='study_judge')
                new_labels = {}
                for result in study_query['results']:
                    st.markdown(f"**{result['rank']}. {result['case_name']}** · {result['doc_id']}")
                    st.write(result['evidence']['text'])
                    with st.expander(f"All original passages · study rank {result['rank']}"):
                        for passage in engine.documents[result['doc_id']].get('passages', []):
                            st.write(passage)
                    saved = study['judgments'].get(selected, {}).get(result['doc_id'], {}).get('label', 'Unjudged')
                    new_labels[result['doc_id']] = st.radio('Study relevance', list(human.LABELS),
                        index=human.LABELS.index(saved), key=f"formal_{study['study_id']}_{selected}_{result['doc_id']}", horizontal=True)
                if not study_query['results']:
                    st.caption('No results returned. Saving confirms human review of this empty list; P@5 will be zero.')
                save = st.form_submit_button('Save study judgments', type='primary')
            if save:
                try:
                    study = human.save_labels(human.STUDY_PATH, study['study_id'], selected, new_labels, judge)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.warning(str(error))
            study_summary = human.summary(study)
            st.dataframe(pd.DataFrame(study_summary['per_query']), hide_index=True, width='stretch')
            if study_summary['mean_P_at_5'] is not None:
                st.metric('Human mean Precision@5', f"{study_summary['mean_P_at_5']:.3f}")
                st.write(f"Relevant: {study_summary['relevant_pairs']} · Not relevant: {study_summary['non_relevant_pairs']}")
            else:
                st.caption('Human mean P@5: pending. Per-query P@5 appears only after that query is fully judged. Denominator is always 5.')
            st.download_button('↓ Persistent study JSON', json.dumps(study, indent=2, ensure_ascii=False), 'human_judgments.json', 'application/json')
            st.download_button('↓ Persistent judgments CSV', human.judgments_csv(study), 'human_judgments.csv', 'text/csv')
            st.download_button('↓ Human evaluation summary', json.dumps(study_summary, indent=2, ensure_ascii=False), 'human_summary.json', 'application/json')

with pipeline_tab:
    st.subheader('A retrieval system with an inspectable experiment loop')
    st.graphviz_chart((ROOT / 'docs/architecture.dot').read_text(encoding='utf-8'))
    st.caption('Scores are computed over the corpus, then literal/metadata eligibility is applied before top-K. RRF combines BM25 and TF-IDF; it is a separate selectable branch.')
    st.markdown('''
    - **Inverted postings and title zones:** term frequency, document frequency, positions, and a 1.3 case-title weight.
    - **Literal positional checks:** a second surface index preserves stopwords and inflections for quoted phrases. Section 30 cannot satisfy a section 302 anchor.
    - **Evidence-aware BM25:** min-max BM25 plus a bounded 0.15 bonus for the fraction of section/article anchors found literally. This is a transparent prototype heuristic.
    - **Reciprocal rank fusion:** BM25 and TF-IDF ranks contribute `1 / (60 + rank)`, avoiding incompatible raw score scales.
    - **Passage retrieval:** IDF-weighted coverage over overlapping 120-word windows; excerpts retain source wording and passage numbers.
    - **Counterfactual inspection:** remove a term, rerun retrieval, and measure winner movement and top-5 overlap. This measures sensitivity, not causal legal importance.
    ''')
    st.caption('Novelty is the combined inspection workflow for this prototype. No claim of a new research algorithm or superiority to professional legal search tools is made.')
    st.code('src/research_engine.py\napp/research_app.py\nsrc/evaluate_research.py\ntests/test_research_engine.py')
    st.caption('AI assistance: OpenAI Codex helped implement and test the enhanced retrieval, experiments, interface, and documentation. Team members should review and explain submitted code.')

st.markdown('''<div class="desk-footer"><span><strong>LegalLens</strong> · The Research Desk · CSD358 / T6</span>
<span>Source: IndicLegalQA v2 · CC BY 4.0 · Case-linked answer passages</span></div>''', unsafe_allow_html=True)
