"""Build the eight-page report from current, verified artifacts, never fixed results.

python build_enhanced_report.py (requires requirements-report.txt).
"""
from pathlib import Path
from html import escape
import hashlib
import json
import xml.etree.ElementTree as ET

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

ROOT = Path(__file__).resolve().parent
NAVY = colors.HexColor('#18364c')
GOLD = colors.HexColor('#b49559')
MUTED = colors.HexColor('#586c7b')
WIDTH = A4[0] - 90
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Cover', fontName='Times-Bold', fontSize=34, leading=39, textColor=NAVY, spaceAfter=15))
styles.add(ParagraphStyle(name='SectionTitle', fontName='Helvetica-Bold', fontSize=17, leading=21, textColor=NAVY, spaceAfter=13))
styles.add(ParagraphStyle(name='Sub', fontName='Helvetica-Bold', fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name='Copy', fontName='Helvetica', fontSize=9.5, leading=13.5, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name='SmallCopy', fontName='Helvetica', fontSize=8, leading=11, textColor=MUTED, spaceAfter=6))
styles.add(ParagraphStyle(name='Cell', fontName='Helvetica', fontSize=8, leading=10.5, textColor=NAVY))
styles.add(ParagraphStyle(name='CodeLine', fontName='Courier', fontSize=8, leading=11, textColor=NAVY, spaceAfter=5))
story, markdown = [], []


def text(value, style='Copy'):
    story.append(Paragraph(escape(str(value)), styles[style]))
    markdown.append(str(value) + '\n')


def section(number, title):
    if number > 1:
        story.append(PageBreak())
    text(f'{number:02d} / {title}', 'SectionTitle')
    markdown[-1] = f'## {number:02d}. {title}\n'


def sub(title, value):
    text(title, 'Sub')
    markdown[-1] = f'### {title}\n'
    text(value)


def table(headers, rows, widths):
    data = [[Paragraph(escape(str(c)), styles['Cell']) for c in row] for row in [headers] + rows]
    item = Table(data, colWidths=widths, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dce8ef')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f1f5f8'), colors.white]),
        ('LINEBELOW', (0, -1), (-1, -1), .5, colors.HexColor('#bdcdd8'))]))
    story.append(item)
    story.append(Spacer(1, 8))
    markdown.append(' | '.join(map(str, headers)))
    markdown.append(' | '.join(['---'] * len(headers)))
    markdown.extend(' | '.join(map(str, r)) for r in rows)
    markdown.append('')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d7e1e8'))
    canvas.line(45, 42, A4[0] - 45, 42)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(45, 28, 'LegalLens / CSD358 / Track T6 / Verified prototype')
    canvas.drawRightString(A4[0] - 45, 28, str(doc.page))
    canvas.restoreState()


def pipeline():
    drawing = Drawing(WIDTH, 238)
    def box(x, y, w, label):
        drawing.add(Rect(x, y, w, 27, rx=4, ry=4, fillColor=colors.HexColor('#edf3f7'), strokeColor=colors.HexColor('#bdcdd8')))
        drawing.add(String(x + w / 2, y + 10, label, textAnchor='middle', fontName='Helvetica', fontSize=7.4, fillColor=NAVY))
    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=GOLD, strokeWidth=1.2))
        if y1 > y2:
            drawing.add(Polygon([x2 - 2.5, y2 + 4, x2 + 2.5, y2 + 4, x2, y2], fillColor=GOLD, strokeColor=None))
    box(0, 207, 248, 'JSON -> case/date groups -> lexical + surface indexes')
    box(270, 207, 232, 'Query -> parser: text / phrases / exclusions / anchors')
    box(30, 166, 125, 'BM25 scores')
    box(196, 166, 125, 'TF-IDF scores')
    box(367, 166, 135, 'Literal / metadata filters')
    arrow(123, 207, 92, 193); arrow(123, 207, 258, 193)
    arrow(386, 207, 92, 193); arrow(386, 207, 258, 193); arrow(386, 207, 435, 193)
    box(20, 122, 147, 'Evidence-aware: BM25 + anchor bonus')
    box(196, 122, 147, 'RRF: BM25 + TF-IDF ranks')
    arrow(92, 166, 92, 149); arrow(92, 166, 270, 149); arrow(258, 166, 270, 149)
    box(20, 78, 323, 'Choose ONE method -> allowed positive-score IDs -> top-K heap')
    arrow(92, 122, 120, 105); arrow(270, 122, 270, 105)
    arrow(435, 166, 335, 105)
    # Direct BM25/TF-IDF remain selectable, alongside derived methods.
    drawing.add(String(182, 113, 'BM25 and TF-IDF are also direct choices', textAnchor='middle', fontName='Helvetica', fontSize=7, fillColor=MUTED))
    box(20, 34, 323, 'Passage windows -> original excerpt / provenance -> term audit / lab')
    arrow(180, 78, 180, 61)
    box(367, 34, 135, 'Separate A / B evaluations')
    arrow(300, 78, 435, 61)
    drawing.add(String(251, 5, 'Ranking Lab: remove a term -> rerun the same retrieval path', textAnchor='middle', fontName='Helvetica', fontSize=8, fillColor=MUTED))
    return drawing


def metrics_chart(rows):
    drawing = Drawing(WIDTH, 158)
    labels = ['BM25', 'TF-IDF', 'RRF', 'Evidence-aware']
    for i, row in enumerate(rows):
        y = 122 - i * 30
        drawing.add(String(0, y + 4, labels[i], fontName='Helvetica', fontSize=8.5, fillColor=NAVY))
        drawing.add(Rect(100, y, 325 * row['MRR@10'], 17, fillColor=GOLD if i == 3 else NAVY, strokeColor=None))
        drawing.add(String(109 + 325 * row['MRR@10'], y + 4, f"{row['MRR@10']:.4f}", fontName='Helvetica', fontSize=8.5, fillColor=MUTED))
    drawing.add(String(100, 5, 'MRR@10; common 0-to-1 scale. Higher is better.', fontName='Helvetica', fontSize=8, fillColor=MUTED))
    return drawing


def main():
    def read(name):
        return json.loads((ROOT / name).read_text(encoding='utf-8'))
    metadata = read('results/research_evaluation.json')
    rows = metadata['metrics']
    grouping = read('results/grouping_diagnostic.json')
    lab = read('results/ranking_lab_example.json')
    failures = read('results/failure_cases.json')
    human = read('results/human_evaluation/human_summary.json')
    tests = ET.parse(ROOT / 'results/test_results.xml').getroot().find('testsuite')
    if int(tests.attrib['failures']) or int(tests.attrib['errors']):
        raise ValueError('Report requires passing tests.')
    if metadata['query_count'] != 2000:
        raise ValueError('Report requires the full benchmark, not a sample.')

    section(1, 'Problem, scope and document representation')
    text('LegalLens', 'Cover')
    text('The Research Desk', 'SectionTitle')
    text('Case search you can inspect, question and reproduce.')
    sub('Problem and Track T6 relevance', 'Can a sparse case-retrieval prototype expose its scoring evidence and let users test ranking sensitivity while preserving literal query constraints? Track T6 concepts include inverted postings, title zones, positional matching, parametric metadata filtering and top-K selection.')
    sub('Document unit', 'The supplied IndicLegalQA JSON has questions, answers, case names and dates. Existing preprocessing groups answers by cleaned (case_name, judgement_date), removes duplicate answer strings within a group and concatenates the remaining answers. A result is a case-linked answer-summary document, not a complete judgment. Excerpts retain supplied wording; the system generates no legal findings.')
    table(['Representation', 'Verified count / scope'], [
        ['Raw QA records', f"{grouping['raw_records']:,}; {grouping['valid_qa_records']:,} have questions and answers"],
        ['Local document groups', f"{metadata['document_count']:,}; {grouping['unique_cleaned_names']:,} cleaned names and seven extra date groups"],
        ['Evaluation queries', f"{metadata['query_count']:,} supplied test queries; one target case per query"],
        ['Upstream description', '1,256 judgments; this count is not a local grouping ground truth'],
        ['Human study', f"{human['query_count']} queries, {human['required_pairs']} frozen result pairs; {human['status']}"],
    ], [170, WIDTH - 170])
    sub('Working scope', 'Default BM25, TF-IDF, evidence-aware BM25 and RRF build from local JSON and require no API keys, network startup downloads, missing pickles or dense assets. MiniLM remains optional and was not executed or reported as reproduced.')
    sub('Claim boundary', 'The contribution is an integrated inspection workflow for this course project. BM25, TF-IDF and RRF are established methods; the bounded anchor bonus is a disclosed heuristic. Retrieval relevance, token coverage and ranking sensitivity do not establish legal correctness, authority or confidence probabilities.')

    section(2, 'Architecture, IR methods and literal constraints')
    story.append(pipeline())
    text('Figure 1. Actual scoring branches. Corpus-wide scores are computed before eligibility filtering; filters still precede top-K. RRF combines BM25 and TF-IDF and is separately selectable. The editable diagram is docs/architecture.dot.', 'SmallCopy')
    table(['Method', 'Purpose', 'Implementation'], [
        ['BM25', 'Main lexical ranker', 'k1=1.5; b=0.75; length normalization; 1.3 multiplier for a term present in the case title'],
        ['TF-IDF', 'Vector-space baseline', 'Sublinear TF, smoothed IDF, L2 normalization and cosine similarity; score arrays aligned by document ID'],
        ['Evidence-aware', 'BM25 extension', 'minmax(BM25) + 0.15 x fraction of literal section/article anchors found; bonus only for positive BM25 scores'],
        ['RRF', 'Established rank fusion', 'Sum 1/(60+rank) over positive BM25 and TF-IDF lists; document ID breaks ties'],
    ], [91, 118, WIDTH - 209])
    text('BM25 term = IDF * tf*(k1+1)/(tf+k1*length_norm) * query_tf * title_weight', 'CodeLine')
    sub('Parser and literal constraints', 'Lexical tokens use case folding, stopword removal and Porter stemming. A separate surface-token index retains stopwords, numbers and inflections. Quoted phrases require adjacency within one original passage or metadata field; -word removes literal matches. Optional strict section/article constraints distinguish 30 from 302. Anchors identify labels and numbers, not statutes or subsection identities.')
    sub('Passage selection and provenance', 'After ranking, overlapping 120-word windows with 60-word stride are scored by matched-token IDF divided by square-root token length. Each displayed excerpt is an original passage substring, with document ID, passage number and word offsets. Metadata categories use heuristic keywords.')

    section(3, 'Explainability and a real Ranking Lab experiment')
    sub('Term-level score audit', 'For each matched stem, the app shows TF, DF, IDF, title weight, positions (first 12) and the numerical BM25 contribution. Their sum reconstructs raw BM25; it does not reconstruct the normalized evidence-aware or RRF score. The automated reconstruction test includes repeated query terms and title weighting.')
    sub('Leave-one-term-out protocol', 'Remove each of up to eight highest-IDF unquoted term groups, preserving quoted phrases and exclusions; rerun the same method and filters. The interactive lab reports top-five winner movement and Jaccard overlap. The saved worked example additionally records both top-five lists and the original winner\'s full new rank.')
    text(f"Real test query {lab['query_id']}: {lab['original_query']}")
    text(f"Method: {lab['method']}; category: all; strict anchors: off. Removed term: {lab['removed_term']}.")
    text('Modified query: ' + lab['modified_query'])
    table(['Rank', 'Original top five', 'New top five'], [
        [i + 1, f"{a['case_name']} ({a['doc_id']})", f"{b['case_name']} ({b['doc_id']})"]
        for i, (a, b) in enumerate(zip(lab['original_top_5'], lab['new_top_5']))
    ], [43, (WIDTH - 43) / 2, (WIDTH - 43) / 2])
    text(f"Winner changed: {lab['winner_changed']}. Original winner's full new rank: {lab['original_winner_new_rank']}. Top-five Jaccard: {lab['top_5_jaccard']:.6f} (4 shared IDs / 6 union IDs). Repeat rankings agree.")
    text(lab['selection'], 'SmallCopy')
    sub('Interpretation', lab['interpretation'] + ' Removing Contract also removes statute context from the question; a new winning criminal case does not answer a legal causality question. JSON/Markdown artifacts preserve the exact experiment. Side-by-side methods and downloadable evidence briefs expose disagreements and provenance.')

    section(4, 'A. Dataset-qrel evaluation')
    text(f"A fresh run evaluates {metadata['query_count']:,} test queries against {metadata['document_count']:,} documents. One supplied target case defines each qrel. P@k uses denominator k, R@k is target-case hit rate, MRR@10 uses reciprocal target rank and nDCG@10 uses 1/log2(rank+1). Metrics are macro-averaged over queries.")
    heads = ['Method', 'P@5', 'R@5', 'P@10', 'R@10', 'MRR@10', 'nDCG@10']
    names = {'BM25': 'BM25', 'TF-IDF': 'TF-IDF', 'Rank fusion (BM25 + TF-IDF)': 'RRF', 'Evidence-aware BM25': 'Evidence-aware'}
    table(heads, [[names[r['Method']]] + [f"{r[h]:.4f}" for h in heads[1:]] for r in rows], [103] + [(WIDTH - 103) / 6] * 6)
    story.append(metrics_chart(rows))
    bm, ea, rrf = rows[0]['MRR@10'], rows[-1]['MRR@10'], rows[2]['MRR@10']
    text(f"Figure 2. Evidence-aware MRR@10 is {ea:.6f} versus BM25 {bm:.6f}: a tiny numerical difference of {ea - bm:+.6f}. Its anchor bonus changes the target rank on {metadata['anchor_rank_changes']} queries; P@5 and R@10 are unchanged. RRF ({rrf:.6f}) performs worse than BM25 here. No paired statistical test was performed; no significance is claimed.")
    sub('Evaluation boundary', 'The corpus was built from all supplied answers before query splitting, including answers associated with test queries. These results measure case lookup over answer summaries, not independent retrieval of unseen full judgments. One-case qrels are incomplete: other relevant cases can be treated as nonrelevant. Anchor bonus 0.15 and RRF k=60 are fixed prototype settings, not tuned on these test queries.')
    text('Recorded run: ' + metadata['created_at'], 'SmallCopy')
    text('Corpus SHA-256: ' + metadata['corpus_sha256'], 'SmallCopy')
    text('Query SHA-256: ' + metadata['query_sha256'], 'SmallCopy')
    text('Historical dense/hybrid artifacts remain explicitly separate and were not newly verified.', 'SmallCopy')

    section(5, 'B. Human-judged evaluation')
    text(human['status'], 'SectionTitle')
    text(f"Implementation ready: {human['query_count']} frozen queries and {human['required_pairs']} top-five pairs. Current progress: {human['judged_pairs']} judged pairs, {human['completed_queries']} completed queries. Human mean P@5: " + (f"{human['mean_P_at_5']:.4f}" if human['mean_P_at_5'] is not None else 'pending.'))
    text('Default queries are selected deterministically in category round-robin order from eligible 7-to-40-word test questions. Categories use the target document only for sampling; neither supplied target IDs nor answers become human labels. This is a small convenience study, not an independent random sample. A custom JSON list of 10-20 queries and another lexical method can be configured through the CLI.')
    text('Judge each case for relevance to the information need using the excerpt and all source passages. Save Relevant / Not Relevant with reviewer initials. Partial labels remain on disk across sessions. Quoted study results and settings are frozen; sidebar search controls do not modify them. Human metrics stay separate from dataset qrels.')
    table(['Query ID', 'Judged / returned', 'Relevant in top five', 'P@5'], [
        [q['query_id'], f"{q['judged']} / {q['returned']}", q['relevant'] if q['complete'] else 'Pending',
         f"{q['P_at_5']:.2f}" if q['complete'] else 'Pending'] for q in human['per_query']
    ], [93, 118, 169, WIDTH - 380])
    text('P@5 = relevant returned results / 5, including shorter lists. A query is complete only after all returned sources have labels; an empty list requires explicit human review. Mean P@5 is displayed only when the whole study is complete. No corpus-wide recall is inferred.', 'SmallCopy')
    text('Persistent artifacts: results/human_evaluation/human_judgments.json and .csv; human_summary.json and .csv. Streamlit exports labels and summary. No human labels have been fabricated.', 'SmallCopy')

    section(6, 'Observed failures and grouping uncertainty')
    for number, failure in enumerate(failures['examples'], 1):
        sub(f'Limitation {number}: ' + failure['query'], 'Intended: ' + failure['intended'])
        text('Observed: ' + failure['observed'])
        text('Alternate query: ' + failure['alternate_query'])
        text('Reason: ' + failure['reason'])
    sub('1,256 upstream judgments versus 1,260 local groups', f"The reproducible raw-data diagnostic finds {grouping['unique_cleaned_names']:,} cleaned names plus {grouping['extra_date_groups']} extra date groups, giving {grouping['local_case_date_groups']:,} groups. No names, dates, questions or answers are missing. There are {grouping['duplicate_complete_records']} repeated complete QA records; repetition does not create extra groups. Two punctuation/case-normalized name collisions are flagged for review, not automatically merged.")
    table(['Same name, multiple dates (examples)', 'Dates stored locally'], [
        [r['case_name'], '; '.join(r['dates'])] for r in grouping['multi_date_names'][:3]
    ], [285, WIDTH - 285])
    text('All seven names and record counts are in grouping_diagnostic.json. The processed groups and per-group QA counts agree with the grouping rule. Dates may describe distinct decisions or contain metadata errors. Without upstream judgment IDs or source URLs, the exact correspondence to 1,256 judgments remains unresolved. Blind merging would change document IDs and qrels without evidence.')
    sub('Remaining limits', 'Stopword removal discards lexical negation such as not; literal phrases preserve it. Literal constraints do not normalize spelling, abbreviations or statutes, and anchors simplify subsection references. Heuristic categories may misclassify cases. Original passage preservation is traceability to the supplied dataset, not authentication of a court holding or current law.')

    section(7, 'Reproducibility and validation')
    text('From Midsem_Assignment with Python 3.10+; activate the created environment before these commands:')
    for command in ['python -m venv .venv', '.\\.venv\\Scripts\\Activate.ps1',
                    'python -m pip install -r requirements-core.txt -r requirements-report.txt pytest',
                    'python -m src.evaluate_research', 'python -m src.submission_evidence',
                    'python -m src.human_evaluation summary',
                    'python -m pytest -q --basetemp=tmp/pytest --junitxml=results/test_results.xml',
                    'python build_enhanced_report.py', 'python -m src.verify_submission']:
        text(command, 'CodeLine')
    text('The supplied study already exists. If absent, use python -m src.human_evaluation prepare first; it refuses to overwrite labels. App: python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1. For Linux/macOS, activate .venv/bin/activate.', 'SmallCopy')
    sub('Artifacts and freshness checks', 'research_metrics.csv, research_evaluation.json and research_per_query.json store aggregates, all 2,000 per-query top-ten lists and scores, configuration, source/input hashes, Python/package versions and the effective stopword hash. Verification recomputes aggregate metrics from saved rankings and checks CSV/JSON agreement, counts, source fingerprints and report facts. --limit runs use separate filenames.')
    text(f"Determinism scope: {metadata['deterministic_repeat_checks']} repeated rankings (first 20 queries x four methods) plus the saved Ranking Lab repeat. Document IDs break score ties. This does not establish determinism across every library/platform version. Record the reported environment when reproducing.")
    sub('Automated validation', f"{tests.attrib['tests']} tests passed; {tests.attrib['failures']} failures and {tests.attrib['errors']} errors in the final recorded run. Test results: results/test_results.xml. Tests use synthetic labels in temporary fixtures only; none enter the real human study.")
    table(['Claim / test family', 'Evidence'], [
        ['Sparse retrieval and arithmetic', 'tests/test_retrieval.py: original preprocessing, indexes and BM25/TF-IDF checks'],
        ['Constraints, score audit, evidence', 'tests/test_research_engine.py: alignment, pre-top-K filters, phrase boundaries, stopwords/inflections, exact numbers, exclusions, reconstruction, original windows, RRF ties, highlighting, metric arithmetic'],
        ['Human study persistence and metrics', 'tests/test_human_evaluation.py: pending state, no auto-labels, reload/merge, fixed denominator, completion/counts, reset, validation, empty-list review, deterministic sampling'],
        ['Streamlit integration', 'tests/test_app.py: query submission, no-match/parser errors, exploratory labels, formal disk labels restored in a new app session'],
    ], [148, WIDTH - 148])
    text('Report build: install requirements-report.txt, then python build_enhanced_report.py. Re-run verification after rebuilding. PDF regeneration checks the eight-page limit and records all report input hashes in report_facts.json.', 'SmallCopy')

    section(8, 'Contributions, AI use, future work and references')
    sub('Verified team contributions required', 'Replace these placeholders with verified team contributions before submission.')
    for number in range(1, 5):
        text(f'[MEMBER NAME {number}] - [Actual contribution]')
    text('Names and roles are not inferred from code or assigned automatically. Team members must review and explain the components they actually contributed.', 'SmallCopy')
    sub('AI-use declaration', 'OpenAI Codex assisted with the enhanced retrieval engine, literal constraints, source windows, term audits, counterfactuals, Streamlit interface, persistent human-study implementation, tests, benchmark reproduction, diagnostics, documentation and report generation. The supplied original project belongs to the original team. AI-generated drafts do not substitute for team review, actual human relevance judgments or verified authorship.')
    sub('Future work', 'Resolve document identity with original judgment IDs and licensed source texts; ingest full judgments with source URLs; normalize statute/abbreviation variants while preserving intent; improve category annotations; collect independent multi-case qrels with multiple reviewers; validate optional dense comparisons; assess query-level uncertainty and significance. Citation/precedent authority is not implemented.')
    sub('References', 'Veningston K and Apratim Mishra (2024). IndicLegalQA Dataset, version 2. DOI 10.17632/gf8n8cnmvc.2; CC BY 4.0. Publisher describes 10,000 QA pairs from 1,256 judgments. Local representation and discrepancies are disclosed above.')
    text('https://data.mendeley.com/datasets/gf8n8cnmvc/2', 'SmallCopy')
    text('Manning, Raghavan and Schutze (2008). Introduction to Information Retrieval. Inverted/positional indexing, vector-space retrieval and evaluation definitions.')
    text('https://nlp.stanford.edu/IR-book/', 'SmallCopy')
    text('Cormack, Clarke and Buttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR. Established reciprocal-rank formula with k=60.')
    text('https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf', 'SmallCopy')
    text('Manual completion remains: human judgments, verified contributions, and the team\'s 5-8 minute live demonstration recording.', 'SmallCopy')

    output = ROOT / 'REPORT_ENHANCED.pdf'
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=57,
                      title='LegalLens - Verified Research Desk', author='LegalLens team - names pending').build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(output).pages)
    if pages > 8:
        raise ValueError(f'Report overflow: {pages} pages. Repair layout before delivery.')
    (ROOT / 'REPORT_ENHANCED.md').write_text('\n'.join(markdown), encoding='utf-8')
    inputs = ['results/research_evaluation.json', 'results/research_metrics.csv', 'results/research_per_query.json',
              'results/ranking_lab_example.json', 'results/failure_cases.json', 'results/grouping_diagnostic.json',
              'results/human_evaluation/human_judgments.json', 'results/human_evaluation/human_summary.json',
              'results/test_results.xml', 'docs/architecture.dot', 'build_enhanced_report.py',
              'app/research_app.py', 'src/human_evaluation.py', 'src/submission_evidence.py',
              'tests/test_research_engine.py', 'tests/test_human_evaluation.py', 'tests/test_app.py']
    facts = dict(metrics=rows, human_summary=human, pages=pages,
                 input_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs},
                 pdf_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (ROOT / 'results/report_facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')
    print(f'{output} ({pages} pages)')


if __name__ == '__main__':
    main()
