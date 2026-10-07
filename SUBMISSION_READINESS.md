# LegalLens submission readiness

Automated implementation and verification are complete. Human relevance judgments,
verified team contributions and the team's live recording still require manual work.
This document answers the twelve requested delivery items. The current report is
`REPORT_ENHANCED.pdf` (eight pages); `AUDIT_AND_PLAN.md` is the earlier broad proposal,
not a list of features all implemented in this narrower request.

## 1. Summary of meaningful changes

- Added a persistent human study with 15 frozen dataset questions and 75 top-five
  pairs. Reviewers explicitly save Relevant / Not Relevant labels with initials.
  Labels survive new app sessions; JSON/CSV and summary exports are available.
- Kept supplied dataset qrels and human judgments separate. Per-query P@5 appears
  only after every returned result is judged; denominator is always five. Mean P@5
  remains pending until the entire study is complete. No human labels were invented.
- Saved a real, repeated Ranking Lab experiment with both ranked lists, removed
  term, winner change, original winner's full new rank, Jaccard and input hashes.
- Captured two actual literal-matching limitations with empty outputs and alternate
  rankings. Both are visible in the app and report.
- Corrected the architecture presentation: scores are computed over the corpus;
  eligibility filtering precedes top-K; RRF is a BM25/TF-IDF branch.
- Fixed a genuine phrase-boundary bug: a quoted phrase can no longer match across
  two separately supplied answers that preprocessing concatenated.
- Reran all 2,000 queries and retained every method's per-query top-ten list and
  metrics. Added settings, runtime/source/input fingerprints and 80 repeated ranks.
- Investigated grouping without altering dates, groups, document IDs or qrels.
- Expanded the passing suite from 26 to 40 tests, preserving the original tests.
- Regenerated a compact report from artifacts, with placeholders rather than
  invented names, conservative claims and a reproducibility verifier.

## 2. Files changed and why

Implementation files:

- `src/human_evaluation.py` — new study preparation, deterministic category sampling,
  frozen lists, atomic persistent labels, completion arithmetic, CSV/JSON exports,
  reviewer/ID validation and a non-overwriting CLI. Needed for real human evaluation.
- `app/research_app.py` — formal persistent judging form, pending status, study
  exports, saved worked/failure examples, separate dataset benchmark label, positive
  phrase example and correct architecture graph. Existing exploratory labels remain.
- `src/research_engine.py` — original-passage/metadata token segments for phrase
  verification. Prevents false phrase hits across concatenated answer boundaries.
- `src/evaluate_research.py` — full per-query rankings/metrics, repeat checks,
  configuration and runtime/code/input hashes. Removes dependence on a tiny saved
  error sample when inspecting reproducibility.
- `src/submission_evidence.py` — reproducible Ranking Lab, failure outputs and raw
  grouping diagnostics. These examples use actual retrieval outputs.
- `src/verify_submission.py` — checks current source/input/artifact hashes, all-query
  metric arithmetic, CSV/JSON consistency, study summary and report fingerprints.
- `build_enhanced_report.py` — artifact-driven eight-page PDF and Markdown builder;
  dynamic results, actual examples, human pending/completion state, contributor
  placeholders, real diagram and report input manifest. Hard-coded result prose removed.
- `requirements-report.txt` — optional ReportLab/pypdf dependencies; default retrieval
  remains free of document-generation and dense-model dependencies.

Tests and configuration:

- `tests/test_human_evaluation.py` — 11 collected tests for no automatic labels,
  persistence, completion/counts, fixed denominator, partial/reset states, empty-list
  review, validation and sampling. Synthetic fixture labels are kept in temporary files.
- `tests/test_app.py` — retains the original app integration and adds disk-label
  restoration in a new Streamlit app session. Isolates study files during testing.
- `tests/test_research_engine.py` — adds cross-passage phrase rejection, explicit
  document-ID RRF ties and known-rank positive nDCG arithmetic.
- `pytest.ini` — restricts discovery to tests, avoiding recursive scans of temporary
  and environment directories.
- `.gitignore` — ignores test temporary files; preserves source/evaluation artifacts.

Documentation:

- `docs/architecture.dot` — editable diagram of actual ranking branches and eligibility.
- `docs/CLAIM_TEST_MAP.md` — maps report claims to named executable tests and records
  what those checks do not establish.
- `README.md` — updated setup/reproduction, persistent judging, grouping findings,
  exact positive examples, limitations and remaining manual tasks.
- `DEMO_SCRIPT.md` — updated seven-minute live walkthrough with reproducible
  examples, correct benchmark statements and actual human judging steps.
- `data/README.md` — grouping evidence and query-split contamination boundary.
- `REPORT_ENHANCED.md` / `REPORT_ENHANCED.pdf` — regenerated report, eight pages.
- `SUBMISSION_READINESS.md` — this complete delivery record and checklist.

Generated artifacts:

- `results/research_metrics.csv`, `research_evaluation.json`, `research_per_query.json`
  — fresh 2,000-query aggregate metrics, provenance/configuration and complete traces.
- `results/ranking_lab_example.json` / `.md` — real worked experiment.
- `results/failure_cases.json` — observed limitations and alternative ranked IDs.
- `results/grouping_diagnostic.json` — counts, all seven date pairs, duplicates and collisions.
- `results/human_evaluation/human_judgments.json` / `.csv`, `human_summary.json` / `.csv`
  — frozen study with **zero real labels**, pending metrics and exportable rows.
- `results/test_results.xml` — final passing 40-test suite.
- `results/report_facts.json` — report facts, input fingerprints, page count and PDF hash.
- `results/submission_verification.json` — PASS for the artifact consistency checks.
- `output/preview/submission/` — rendered report pages and a real pending-study screenshot.

The existing BM25/TF-IDF implementations, raw/processed corpus and supplied qrels
were preserved. Optional dense code and historical results remain separate. No new
statute-normalization, precedent authority, mandatory dense model or external API
was introduced in this scope.

## 3. Final real pipeline

1. IndicLegalQA JSON → cleaned (case, date) groups → unique source answer passages.
2. Build stemmed inverted postings/title zones and surface-token postings/segments.
3. Parse the query into scoring text, required phrases, exclusions and anchors.
4. Compute corpus-wide scores for the selected method: raw BM25, TF-IDF,
   Evidence-aware BM25, or RRF of BM25/TF-IDF ranks.
5. Resolve phrase/exclusion/category/optional strict-anchor eligibility; intersect
   with positive-score IDs. This happens before top-K selection.
6. Select top-K through a heap with document-ID tie breaking.
7. Choose a source passage window; preserve text, document ID, passage number and
   word offsets.
8. Inspect raw BM25 contributions, compare methods, rerun term removals or manually
   judge the frozen human study. Evaluation A and evaluation B remain separate.

`docs/architecture.dot` and report Figure 1 show these branches. RRF is neither a
downstream step after Evidence-aware nor mandatory for every query.

## 4. Test results and warnings

**40 passed; 0 failed; 0 errors**, final run 10.50 seconds. All 26 original tests
remain; the expanded suite includes unit and Streamlit integration checks. The
executed record is `results/test_results.xml`; claim mapping is in
`docs/CLAIM_TEST_MAP.md`. No Python warning summary was emitted in the final run.

Windows initially blocked pytest temporary-folder access. The final approved run
used project-local fixtures and scoped test discovery. These setup failures were
not hidden by deleting tests. PDF rendering emitted `No display font for 'Symbol'`;
all eight final rendered pages were visually inspected, with no missing visible
glyphs, clipping, table overflow or overlapping text.

Actual app browser check: `"rent control act"` returned four cases; the formal
judging desk displayed **Human evaluation pending**, 0/15 queries and 0/75 pairs.
No real labels were entered during verification. A new app session's disk label
restoration was tested only with synthetic temporary fixtures.

## 5. Final verified dataset-qrel metrics

All values are from the new full run, using 2,000 test queries and 1,260 documents.
The exact values below avoid display-rounding ambiguity:

- **BM25:** P@5 0.1647; R@5 0.8235; P@10 0.08565; R@10 0.8565;
  MRR@10 0.7711738095238095; nDCG@10 0.791743201279914.
- **TF-IDF:** P@5 0.1625; R@5 0.8125; P@10 0.08495; R@10 0.8495;
  MRR@10 0.7515936507936508; nDCG@10 0.7752138471063684.
- **RRF:** P@5 0.1646; R@5 0.8230; P@10 0.08560; R@10 0.8560;
  MRR@10 0.7622922619047618; nDCG@10 0.7849205313387455.
- **Evidence-aware BM25:** P@5 0.1647; R@5 0.8235; P@10 0.08565; R@10 0.8565;
  MRR@10 0.7714654761904762; nDCG@10 0.7919623981240916.

Evidence-aware MRR differs by **+0.000291666667**, with four changed target ranks.
P@5 and R@10 are unchanged. **RRF is worse than BM25 here.** No paired statistical
test was performed, so no significance claim is made. Tiny binary floating-point
representations in CSV/JSON do not indicate a metric disagreement.

All supplied answers, including test-linked answers, are in the indexed corpus.
This is answer-summary case lookup with one-case qrels, not independent retrieval
over unseen full judgments or a corpus-wide human relevance benchmark.

## 6. Human evaluation status

**Implementation ready, judgments still pending.**

- Study: 15 frozen dataset questions; Evidence-aware BM25; top five; 75 result pairs.
- Actual labels: zero. Completed queries: zero.
- Human per-query P@5 and mean P@5: **pending / null**, not fabricated zero scores.
- Humans save labels with initials through **Judge relevance → B. Human-judged evaluation**.
- Partial labels persist across sessions. Per-query precision requires every returned
  result; the mean requires all queries. Denominator is always five.
- JSON/CSV labels and JSON/CSV summaries are persisted locally and exportable.
- Configurable 10–20-query JSON lists and offline lexical methods are supported by CLI.

This is a convenience sample across heuristic categories, not independent random
sampling. Some supplied questions are context-dependent. Humans should judge the
stated question, or prepare a clearer custom study before beginning labels. Use one
judging process; concurrent writers and inter-reviewer agreement are not implemented.

## 7. Real Ranking Lab example

Dataset query **q_05836**:

> What is the significance of Section 25(3) of the Indian Contract Act in this case?

Method: Evidence-aware BM25; category all; strict anchors off; top-five inspection.
Removed term: **Contract**. Modified query:

> What is the significance of Section 25(3) of the Indian Act in this case?

Original top five:

1. K. Hymavathi vs. The State of Andhra Pradesh & Anr. — doc_0397
2. Sudhir Kumar Jain vs. State of Rajasthan — doc_1043
3. Ashoksinh Jayendrasinh vs. State of Gujarat — doc_0075
4. Nasiruddin & Anr. vs. The State of Uttar Pradesh — doc_0643
5. Anwar @ Bhugra vs State of Haryana — doc_0051

New top five:

1. Sudhir Kumar Jain vs. State of Rajasthan — doc_1043
2. Ashoksinh Jayendrasinh vs. State of Gujarat — doc_0075
3. Anwar @ Bhugra vs State of Haryana — doc_0051
4. K. Hymavathi vs. The State of Andhra Pradesh & Anr. — doc_0397
5. Bhoopendra Singh vs. State of Rajasthan & Anr. — doc_0125

Winner changed: yes. Original winner's full new rank: **4**. Jaccard: **4/6 = 0.666667**.
Both original and modified rankings were repeated and agreed. The fixed query and
first winner-changing removal were selected for an explanation demonstration, not
to estimate accuracy. Removing a statute-context word demonstrates sensitivity;
it does not establish legal importance, causality, authority or correctness.

## 8. Observed failure cases and grouping findings

**Literal full-name limitation:** `"Hindu Adoptions and Maintenance Act" section 12`
was intended to find passages about the named Act/section. It returns **zero**,
because the complete required surface phrase is absent. `adoption maintenance
section 12` returns five lexical candidates. Their legal relevance is not asserted.

**Abbreviation limitation:** `"sec. 302"` was intended as a variant of section 302.
It returns **zero**; `"section 302"` returns five, beginning with doc_1049. Surface
matching does not expand sec to section. Actual ranked IDs and parsed plans are
saved in `results/failure_cases.json`.

**Grouping:** the supplied raw data has **1,253 cleaned names + seven extra date
groups = 1,260 local groups**. There are zero missing names/dates/questions/answers,
14 duplicate complete QA records and two normalized-name collision pairs. Existing
processed keys/QA counts agree with the rule. The publisher reports 1,256 judgments.
Same-name different-date records could be distinct decisions or metadata errors;
no original judgment IDs/URLs establish their identity. The exact upstream mapping
remains unresolved. No dates or document groups were changed.

## 9. Reproduction from a clean project environment

Open PowerShell in the extracted `Midsem_Assignment` directory with Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt -r requirements-report.txt pytest
.\.venv\Scripts\python.exe -m src.evaluate_research
.\.venv\Scripts\python.exe -m src.submission_evidence
.\.venv\Scripts\python.exe -m src.human_evaluation summary
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest --junitxml=results/test_results.xml
.\.venv\Scripts\python.exe build_enhanced_report.py
.\.venv\Scripts\python.exe -m src.verify_submission
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The supplied study already exists. If missing, run `python -m src.human_evaluation
prepare` before `summary`. Preparation refuses to overwrite labels. For a custom
list: `python -m src.human_evaluation prepare --queries config/my_queries.json
--method BM25 --output results/human_bm25/human_judgments.json`. The list contains
10–20 `{query_id, query}` objects; the default UI reads the default study path.

Package installation may need internet; default application startup and retrieval
do not. The actual execution used the prepared local environment; its versions are
recorded in `research_evaluation.json`. A second newly installed virtual environment
was not benchmarked. Arbitrary future library versions are not claimed identical.

Corpus SHA-256:
`6e7e07d2230f3a779fc3b82bc2508eef1dd48dbc6465a5f14dca3910d2bdf714`

Query SHA-256:
`427b2e18d697671c01064a092ba5351c5fa658e274bbd823725ade9388a826fd`

Sample evaluation runs write separate files. Verifier checks 2,000-query counts,
source/input fingerprints, rankings-derived aggregates, CSV/JSON agreement, study
status and report hashes. Determinism checks cover 20 queries × four methods plus
the worked example, not every possible query/platform. After changing labels or
regenerating tests/results, rebuild the report before verification.

## 10. Remaining manual tasks

1. Complete the actual 75 human query-result judgments, or deliberately choose a
   new custom study before judging. Save/export real labels; regenerate summary and
   report, then verify again. Do not copy qrels or test fixture labels as human work.
2. Replace all four `[MEMBER NAME n] - [Actual contribution]` entries with verified
   names and actual contributions. The builder contains these placeholders; update
   it too so a later regeneration preserves the verified text.
3. Record the team's 5–8 minute live demo using `DEMO_SCRIPT.md`; explain actual
   code, metrics and limitations. The screenshot is verification evidence, not a video.
4. Review the AI-use declaration and source credit, rehearse component ownership,
   and upload the final code/report/video using the course's submission procedure.

If labels are still pending, retain that exact status in the submission. Resolving
upstream judgment identity requires source evidence; it is not a guessed manual merge.

## 11. Report changes

Regenerated eight-page PDF and Markdown contain:

1. Problem, T6 concepts, grouped-answer document unit and scope boundaries.
2. Correct architecture, method comparison, parser/constraints and passage provenance.
3. Raw score audit and the real worked Ranking Lab example, including both top-five lists.
4. Separate dataset-qrel evaluation, all six verified metrics, tiny gain and RRF loss.
5. Separate human evaluation protocol, all 15 pending rows, fixed denominator and exports.
6. Two actual failure demonstrations, grouping diagnostic and remaining limitations.
7. Reproduction commands, artifact freshness, repeat-check scope and actual test families.
8. Four contribution placeholders, AI-use declaration, future work and references.

All eight final pages were rendered and inspected. `report_facts.json` confirms
page count and numeric agreement; `submission_verification.json` reports PASS.
The optional paired statistical test is not implemented or implied.

## 12. Final submission checklist

- **PASS** — existing sparse project preserved; no restart or replacement of rankers.
- **PASS** — working offline JSON-based default without API keys, dense assets or pickles.
- **PASS** — literal phrase boundary bug fixed; filtering remains before top-K.
- **PASS** — 40 automated tests; unit and Streamlit integration checks.
- **PASS** — actual full 2,000-query evaluation and complete per-query artifact.
- **PASS** — counts, hashes, sampled deterministic repeats and report agreement verified.
- **PASS** — persistent human judging, fixed P@5, progress and exports implemented.
- **PASS** — real Ranking Lab and failure examples saved and shown in report/UI.
- **PASS** — grouping diagnostic reproducible; uncertainty disclosed; corpus unchanged.
- **PASS** — eight-page artifact-driven report, method summary, architecture, AI/source credit.
- **NEEDS MANUAL ACTION** — real human judgments and resulting mean P@5.
- **NEEDS MANUAL ACTION** — verified member names and contributions.
- **NEEDS MANUAL ACTION** — team demo/video and course submission.
- **NOT IMPLEMENTED (optional)** — paired statistical significance test.
- **NOT IMPLEMENTED** — authoritative full-judgment source identity reconciliation,
  legal authority scoring, statute/abbreviation normalization and inter-reviewer agreement.
- **NOT IMPLEMENTED (optional evaluation)** — new MiniLM/dense reproduction;
  optional code is preserved, assets are absent and historical scores stay separate.

The last optional item is not a failure of the submitted sparse default. Automated
readiness does not mean human evaluation or verified authorship has been completed.
