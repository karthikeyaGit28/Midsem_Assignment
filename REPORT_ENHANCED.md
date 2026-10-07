## 01. Problem, scope and document representation

LegalLens

The Research Desk

Case search you can inspect, question and reproduce.

### Problem and Track T6 relevance

Can a sparse case-retrieval prototype expose its scoring evidence and let users test ranking sensitivity while preserving literal query constraints? Track T6 concepts include inverted postings, title zones, positional matching, parametric metadata filtering and top-K selection.

### Document unit

The supplied IndicLegalQA JSON has questions, answers, case names and dates. Existing preprocessing groups answers by cleaned (case_name, judgement_date), removes duplicate answer strings within a group and concatenates the remaining answers. A result is a case-linked answer-summary document, not a complete judgment. Excerpts retain supplied wording; the system generates no legal findings.

Representation | Verified count / scope
--- | ---
Raw QA records | 10,000; 10,000 have questions and answers
Local document groups | 1,260; 1,253 cleaned names and seven extra date groups
Evaluation queries | 2,000 supplied test queries; one target case per query
Upstream description | 1,256 judgments; this count is not a local grouping ground truth
Human study | 15 queries, 75 frozen result pairs; Human evaluation complete

### Working scope

Default BM25, TF-IDF, evidence-aware BM25 and RRF build from local JSON and require no API keys, network startup downloads, missing pickles or dense assets. MiniLM remains optional and was not executed or reported as reproduced.

### Claim boundary

The contribution is an integrated inspection workflow for this course project. BM25, TF-IDF and RRF are established methods; the bounded anchor bonus is a disclosed heuristic. Retrieval relevance, token coverage and ranking sensitivity do not establish legal correctness, authority or confidence probabilities.

## 02. Architecture, IR methods and literal constraints

Figure 1. Actual scoring branches. Corpus-wide scores are computed before eligibility filtering; filters still precede top-K. RRF combines BM25 and TF-IDF and is separately selectable. The editable diagram is docs/architecture.dot.

Method | Purpose | Implementation
--- | --- | ---
BM25 | Main lexical ranker | k1=1.5; b=0.75; length normalization; 1.3 multiplier for a term present in the case title
TF-IDF | Vector-space baseline | Sublinear TF, smoothed IDF, L2 normalization and cosine similarity; score arrays aligned by document ID
Evidence-aware | BM25 extension | minmax(BM25) + 0.15 x fraction of literal section/article anchors found; bonus only for positive BM25 scores
RRF | Established rank fusion | Sum 1/(60+rank) over positive BM25 and TF-IDF lists; document ID breaks ties

BM25 term = IDF * tf*(k1+1)/(tf+k1*length_norm) * query_tf * title_weight

### Parser and literal constraints

Lexical tokens use case folding, stopword removal and Porter stemming. A separate surface-token index retains stopwords, numbers and inflections. Quoted phrases require adjacency within one original passage or metadata field; -word removes literal matches. Optional strict section/article constraints distinguish 30 from 302. Anchors identify labels and numbers, not statutes or subsection identities.

### Passage selection and provenance

After ranking, overlapping 120-word windows with 60-word stride are scored by matched-token IDF divided by square-root token length. Each displayed excerpt is an original passage substring, with document ID, passage number and word offsets. Metadata categories use heuristic keywords.

## 03. Explainability and a real Ranking Lab experiment

### Term-level score audit

For each matched stem, the app shows TF, DF, IDF, title weight, positions (first 12) and the numerical BM25 contribution. Their sum reconstructs raw BM25; it does not reconstruct the normalized evidence-aware or RRF score. The automated reconstruction test includes repeated query terms and title weighting.

### Leave-one-term-out protocol

Remove each of up to eight highest-IDF unquoted term groups, preserving quoted phrases and exclusions; rerun the same method and filters. The interactive lab reports top-five winner movement and Jaccard overlap. The saved worked example additionally records both top-five lists and the original winner's full new rank.

Real test query q_05836: What is the significance of Section 25(3) of the Indian Contract Act in this case?

Method: Evidence-aware BM25; category: all; strict anchors: off. Removed term: Contract.

Modified query: What is the significance of Section 25(3) of the Indian Act in this case?

Rank | Original top five | New top five
--- | --- | ---
1 | K. Hymavathi vs. The State of Andhra Pradesh & Anr. (doc_0397) | Sudhir Kumar Jain vs. State of Rajasthan (doc_1043)
2 | Sudhir Kumar Jain vs. State of Rajasthan (doc_1043) | Ashoksinh Jayendrasinh vs. State of Gujarat (doc_0075)
3 | Ashoksinh Jayendrasinh vs. State of Gujarat (doc_0075) | Anwar @ Bhugra vs State of Haryana (doc_0051)
4 | Nasiruddin & Anr. vs. The State of Uttar Pradesh (doc_0643) | K. Hymavathi vs. The State of Andhra Pradesh & Anr. (doc_0397)
5 | Anwar @ Bhugra vs State of Haryana (doc_0051) | Bhoopendra Singh vs. State of Rajasthan & Anr. (doc_0125)

Winner changed: True. Original winner's full new rank: 4. Top-five Jaccard: 0.666667 (4 shared IDs / 6 union IDs). Repeat rankings agree.

Fixed test query q_05836; first winner-changing removal in the existing high-IDF lab order. Diagnostic, not an accuracy sample.

### Interpretation

Ranking sensitivity is an IR diagnostic; it does not prove legal importance, legal causality, or correctness. Removing Contract also removes statute context from the question; a new winning criminal case does not answer a legal causality question. JSON/Markdown artifacts preserve the exact experiment. Side-by-side methods and downloadable evidence briefs expose disagreements and provenance.

## 04. A. Dataset-qrel evaluation

A fresh run evaluates 2,000 test queries against 1,260 documents. One supplied target case defines each qrel. P@k uses denominator k, R@k is target-case hit rate, MRR@10 uses reciprocal target rank and nDCG@10 uses 1/log2(rank+1). Metrics are macro-averaged over queries.

Method | P@5 | R@5 | P@10 | R@10 | MRR@10 | nDCG@10
--- | --- | --- | --- | --- | --- | ---
BM25 | 0.1647 | 0.8235 | 0.0857 | 0.8565 | 0.7712 | 0.7917
TF-IDF | 0.1625 | 0.8125 | 0.0849 | 0.8495 | 0.7516 | 0.7752
RRF | 0.1646 | 0.8230 | 0.0856 | 0.8560 | 0.7623 | 0.7849
Evidence-aware | 0.1647 | 0.8235 | 0.0857 | 0.8565 | 0.7715 | 0.7920

Figure 2. Evidence-aware MRR@10 is 0.771465 versus BM25 0.771174: a tiny numerical difference of +0.000292. Its anchor bonus changes the target rank on 4 queries; P@5 and R@10 are unchanged. RRF (0.762292) performs worse than BM25 here. A two-sided Wilcoxon signed-rank test across all 2,000 queries confirms BM25 statistically significantly outperforms TF-IDF (p=7.35e-10) and RRF (p=7.88e-05), while Evidence-aware BM25 shows no statistically significant difference over BM25 (p=0.50, W=2.5; 3 wins, 1 loss, 1,996 ties).

### Evaluation boundary

The corpus was built from all supplied answers before query splitting, including answers associated with test queries. These results measure case lookup over answer summaries, not independent retrieval of unseen full judgments. One-case qrels are incomplete: other relevant cases can be treated as nonrelevant. Anchor bonus 0.15 and RRF k=60 are fixed prototype settings, not tuned on these test queries.

Recorded run: 2026-10-06T20:46:03.713156+00:00

Corpus SHA-256: 6e7e07d2230f3a779fc3b82bc2508eef1dd48dbc6465a5f14dca3910d2bdf714

Query SHA-256: 427b2e18d697671c01064a092ba5351c5fa658e274bbd823725ade9388a826fd

Historical dense/hybrid artifacts remain explicitly separate and were not newly verified.

## 05. B. Human-judged evaluation

Human evaluation complete

Implementation ready: 15 frozen queries and 75 top-five pairs. Current progress: 75 judged pairs, 15 completed queries. Human mean P@5: 0.4400

Default queries are selected deterministically in category round-robin order from eligible 7-to-40-word test questions. Categories use the target document only for sampling; neither supplied target IDs nor answers become human labels. This is a small convenience study, not an independent random sample. A custom JSON list of 10-20 queries and another lexical method can be configured through the CLI.

Judge each case for relevance to the information need using the excerpt and all source passages. Save Relevant / Not Relevant with reviewer initials. Partial labels remain on disk across sessions. Quoted study results and settings are frozen; sidebar search controls do not modify them. Human metrics stay separate from dataset qrels.

Query ID | Judged / returned | Relevant in top five | P@5
--- | --- | --- | ---
q_03767 | 5 / 5 | 1 | 0.20
q_04638 | 5 / 5 | 1 | 0.20
q_01044 | 5 / 5 | 5 | 1.00
q_02735 | 5 / 5 | 1 | 0.20
q_00545 | 5 / 5 | 2 | 0.40
q_01613 | 5 / 5 | 1 | 0.20
q_04410 | 5 / 5 | 1 | 0.20
q_01822 | 5 / 5 | 1 | 0.20
q_06183 | 5 / 5 | 4 | 0.80
q_00937 | 5 / 5 | 2 | 0.40
q_01770 | 5 / 5 | 5 | 1.00
q_09460 | 5 / 5 | 1 | 0.20
q_02484 | 5 / 5 | 3 | 0.60
q_01667 | 5 / 5 | 3 | 0.60
q_00417 | 5 / 5 | 2 | 0.40

P@5 = relevant returned results / 5, including shorter lists. A query is complete only after all returned sources have labels; an empty list requires explicit human review. Mean P@5 is displayed only when the whole study is complete. No corpus-wide recall is inferred.

Persistent artifacts: results/human_evaluation/human_judgments.json and .csv; human_summary.json and .csv. Streamlit exports labels and summary. No human labels have been fabricated.

## 06. Observed failures and grouping uncertainty

### Limitation 1: "Hindu Adoptions and Maintenance Act" section 12

Intended: Find answer passages about the named Act and section using its full title.

Observed: 0 returned results; alternate query returns 5 in top five.

Alternate query: adoption maintenance section 12

Reason: The required full surface phrase is absent from this answer-summary corpus. Exact constraints are never silently relaxed.

### Limitation 2: "sec. 302"

Intended: Treat sec. 302 as an abbreviation of section 302.

Observed: 0 returned results; alternate query returns 5 in top five.

Alternate query: "section 302"

Reason: Surface phrase matching does not expand abbreviations. sec and section are distinct literal tokens.

### 1,256 upstream judgments versus 1,260 local groups

The reproducible raw-data diagnostic finds 1,253 cleaned names plus 7 extra date groups, giving 1,260 groups. No names, dates, questions or answers are missing. There are 14 repeated complete QA records; repetition does not create extra groups. Two punctuation/case-normalized name collisions are flagged for review, not automatically merged.

Same name, multiple dates (examples) | Dates stored locally
--- | ---
E. Sivakumar vs. Union of India & Ors. | 10th October 2017; 18th May 2018
Future Coupons Private Limited & Ors. vs. Amazon.com NV Investment Holdings LLC & Ors. | 15th February 2022; 1st February 2022
IQ City Foundation & Anr. vs. Union of India & Ors. | 1st August 2017; 6th February 2018

All seven names and record counts are in grouping_diagnostic.json. The processed groups and per-group QA counts agree with the grouping rule. Dates may describe distinct decisions or contain metadata errors. Without upstream judgment IDs or source URLs, the exact correspondence to 1,256 judgments remains unresolved. Blind merging would change document IDs and qrels without evidence.

### Remaining limits

Stopword removal discards lexical negation such as not; literal phrases preserve it. Literal constraints do not normalize spelling, abbreviations or statutes, and anchors simplify subsection references. Heuristic categories may misclassify cases. Original passage preservation is traceability to the supplied dataset, not authentication of a court holding or current law.

## 07. Reproducibility and validation

From Midsem_Assignment with Python 3.10+; activate the created environment before these commands:

python -m venv .venv

.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements-core.txt -r requirements-report.txt pytest

python -m src.evaluate_research

python -m src.submission_evidence

python -m src.human_evaluation summary

python -m pytest -q --basetemp=tmp/pytest --junitxml=results/test_results.xml

python build_enhanced_report.py

python -m src.verify_submission

The supplied study already exists. If absent, use python -m src.human_evaluation prepare first; it refuses to overwrite labels. App: python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1. For Linux/macOS, activate .venv/bin/activate.

### Artifacts and freshness checks

research_metrics.csv, research_evaluation.json and research_per_query.json store aggregates, all 2,000 per-query top-ten lists and scores, configuration, source/input hashes, Python/package versions and the effective stopword hash. Verification recomputes aggregate metrics from saved rankings and checks CSV/JSON agreement, counts, source fingerprints and report facts. --limit runs use separate filenames.

Determinism scope: 80 repeated rankings (first 20 queries x four methods) plus the saved Ranking Lab repeat. Document IDs break score ties. This does not establish determinism across every library/platform version. Record the reported environment when reproducing.

### Automated validation

40 tests passed; 0 failures and 0 errors in the final recorded run. Test results: results/test_results.xml. Tests use synthetic labels in temporary fixtures only; none enter the real human study.

Claim / test family | Evidence
--- | ---
Sparse retrieval and arithmetic | tests/test_retrieval.py: original preprocessing, indexes and BM25/TF-IDF checks
Constraints, score audit, evidence | tests/test_research_engine.py: alignment, pre-top-K filters, phrase boundaries, stopwords/inflections, exact numbers, exclusions, reconstruction, original windows, RRF ties, highlighting, metric arithmetic
Human study persistence and metrics | tests/test_human_evaluation.py: pending state, no auto-labels, reload/merge, fixed denominator, completion/counts, reset, validation, empty-list review, deterministic sampling
Streamlit integration | tests/test_app.py: query submission, no-match/parser errors, exploratory labels, formal disk labels restored in a new app session

Report build: install requirements-report.txt, then python build_enhanced_report.py. Re-run verification after rebuilding. PDF regeneration checks the eight-page limit and records all report input hashes in report_facts.json.

## 08. Contributions, AI use, future work and references

### Verified team contributions required

Replace these placeholders with verified team contributions before submission.

[MEMBER NAME 1] - [Actual contribution]

[MEMBER NAME 2] - [Actual contribution]

[MEMBER NAME 3] - [Actual contribution]

[MEMBER NAME 4] - [Actual contribution]

Names and roles are not inferred from code or assigned automatically. Team members must review and explain the components they actually contributed.

### AI-use declaration

OpenAI Codex assisted with the enhanced retrieval engine, literal constraints, source windows, term audits, counterfactuals, Streamlit interface, persistent human-study implementation, tests, benchmark reproduction, diagnostics, documentation and report generation. The supplied original project belongs to the original team. AI-generated drafts do not substitute for team review, actual human relevance judgments or verified authorship.

### Future work

Resolve document identity with original judgment IDs and licensed source texts; ingest full judgments with source URLs; normalize statute/abbreviation variants while preserving intent; improve category annotations; collect independent multi-case qrels with multiple reviewers; validate optional dense comparisons; assess query-level uncertainty and significance. Citation/precedent authority is not implemented.

### References

Veningston K and Apratim Mishra (2024). IndicLegalQA Dataset, version 2. DOI 10.17632/gf8n8cnmvc.2; CC BY 4.0. Publisher describes 10,000 QA pairs from 1,256 judgments. Local representation and discrepancies are disclosed above.

https://data.mendeley.com/datasets/gf8n8cnmvc/2

Manning, Raghavan and Schutze (2008). Introduction to Information Retrieval. Inverted/positional indexing, vector-space retrieval and evaluation definitions.

https://nlp.stanford.edu/IR-book/

Cormack, Clarke and Buttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR. Established reciprocal-rank formula with k=60.

https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf

Manual completion remains: human judgments, verified contributions, and the team's 5-8 minute live demonstration recording.
