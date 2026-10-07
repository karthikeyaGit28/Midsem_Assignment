# LegalLens - The Research Desk

A working CSD358 Track T6 information retrieval prototype over case-linked IndicLegalQA answer passages. The enhancement makes rankings **inspectable and testable**: exact source excerpts, literal section/phrase checks, a term-level BM25 audit, and a leave-one-term-out Ranking Lab.

## Run locally

From `Midsem_Assignment` with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

On this prepared Windows checkout, `.venv` is already installed. Run `./run.ps1` or the last command above. Open http://127.0.0.1:8501. For macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

The default app builds BM25, TF-IDF, and literal positional indexes from the shipped `data/processed/documents.json`. No missing pickle files, model downloads, API keys, or NLTK downloads are needed. First startup builds and caches indexes; subsequent searches reuse them. The legacy dense modules remain available separately.

## What to demonstrate

1. Click **Tenancy & notice**, inspect a real source excerpt, and expand its BM25 term contribution table.
2. Open **Ranking Lab**. Compare BM25, TF-IDF, and reciprocal rank fusion, then run the leave-one-term-out experiment. Each row reports the new winning case, original winner's rank within the new top 5, and top-5 Jaccard overlap.
3. Click **Exact source phrase** (`"rent control act"`), then search `"section 302" IPC` with **Require every section / article anchor**. Quoted phrases always constrain results; strict anchors are optional. Phrases cannot cross original answer boundaries.
4. Search `tenant rent -tax` to demonstrate exclusion through literal postings. Source categories are keyword heuristics and can be wrong.
5. Export an **Evidence brief** and **Experiment JSON**. Excerpts carry document IDs, passage numbers, and word ranges; nothing is generated as a judicial holding.
6. In **Judge relevance**, use **B. Human-judged evaluation** for the formal 15-query, 75-pair study. Enter reviewer initials, label each frozen top-five result, and save. Labels persist on disk across sessions. Per-query P@5 uses denominator 5; the mean stays pending until every query is complete. The exploratory current-search labels remain a separate session-only feature.
7. Open **Benchmarks** and show a real limitation alongside the measured results. The novelty is the inspection workflow, not a claim that these well-known scoring algorithms are new.

## IR methods and code

- `src/indexing.py`: stemmed inverted postings, document frequency, token positions, case-name zones.
- `src/bm25_retriever.py`: BM25 with k1=1.5, b=0.75 and a 1.3 case-title term weight.
- `src/tfidf_retriever.py`: sublinear TF-IDF, L2 normalization, cosine similarity.
- `src/research_engine.py`: query parser, literal positional postings, rare-term-first intersection, exclusions, category prefilter, top-K heap, score audits, passage windows, and counterfactuals.
- Evidence-aware score: `minmax(BM25) + 0.15 * fraction_of_literal_anchors_matched`. Only positive lexical matches receive a bonus. This fixed prototype setting was not tuned on the test set; scores are not probabilities.
- Rank fusion: sum `1/(60 + rank)` for BM25 and TF-IDF lists with positive scores. Document IDs break ties deterministically. RRF is an established method [Cormack et al., 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).
- Quoted phrases use literal tokens, retaining stopwords and inflections. They cannot match a stemmed near-phrase. `section 30` cannot satisfy a `section 302` anchor.
- Passage extraction ranks overlapping 120-word source windows with IDF-weighted query-token coverage and a length penalty. It can find evidence after the start of a long passage.
- Counterfactuals remove each of up to eight high-IDF unquoted term groups while keeping quoted phrases and `-word` exclusions. This is ranking sensitivity, not causal legal importance.
- `app/research_app.py`: research interface; `app/streamlit_app.py` is the stable launcher.

## Reproduce evaluation and tests

```powershell
.\.venv\Scripts\python.exe -m src.evaluate_research
.\.venv\Scripts\python.exe -m src.submission_evidence
.\.venv\Scripts\python.exe -m src.human_evaluation summary
.\.venv\Scripts\python.exe -m src.significance_testing
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q --basetemp=tmp/pytest --junitxml=results/test_results.xml
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe build_enhanced_report.py
.\.venv\Scripts\python.exe -m src.verify_submission
```

Evaluation uses all 2,000 supplied test queries and 1,260 case-linked documents. `results/research_metrics.csv` contains freshly computed P@5, R@5, P@10, R@10, MRR@10 and nDCG@10. `results/research_evaluation.json` records input/source SHA-256 hashes, counts, settings, runtime/package versions and 80 repeated-ranking checks. `research_per_query.json` retains all per-query top-ten lists and metrics. `--limit 100` writes separate sample artifacts and cannot overwrite the full benchmark.

BM25 MRR@10 is 0.771174; evidence-aware BM25 is 0.771465. This is a tiny numerical gain of 0.000292; the paired Wilcoxon test finds no significant difference (p=0.50). Statistical artifacts include paired t-tests, Holm correction, and input fingerprints. Rank fusion is worse at 0.762292. These observations are disclosed rather than presented as universal improvements. Historical dense/hybrid results are retained in `results/metrics.csv` and explicitly labelled as original-team artifacts in the app.

The final suite has 48 passing tests, including phrase boundaries, explicit RRF ties, score reconstruction, fixed-denominator human precision, pending/partial states, label validation and persistence into a new Streamlit session. `docs/CLAIM_TEST_MAP.md` maps report claims to actual tests. Windows sandbox restrictions required an approved test run; temporary fixtures are under ignored `tmp/`, not the real human study.

## Persistent human evaluation

**The saved study is complete: 75 labels, 33 relevant and 42 non-relevant, mean P@5 = 0.44.** Reviewer initials are recorded. The team should confirm how these judgments were collected; artifact verification checks arithmetic and freshness. The supplied study has 15 deterministically selected dataset questions across heuristic categories and frozen Evidence-aware BM25 top-five lists. No qrels or generated answers are copied into human labels. This small convenience sample is not an independent relevance benchmark, and some supplied questions need context. Judge their stated information need; choose a custom query set if the team wants clearer independent questions.

Use **Judge relevance → B. Human-judged evaluation**. Enter actual reviewer initials, inspect source passages, mark **Relevant** or **Not Relevant**, and click **Save study judgments**. The app exports JSON, CSV and a summary, and writes them to `results/human_evaluation/`. When incomplete, the mean is `null` and the UI says **Human evaluation pending**. When complete, the summary reports per-query and mean P@5, judged pairs, relevant and non-relevant counts. Recall is not inferred.

If no study exists, run `python -m src.human_evaluation prepare`. It refuses to overwrite an existing study. To create another configuration without discarding labels:

```powershell
python -m src.human_evaluation prepare --queries config/my_queries.json --method BM25 --output results/human_bm25/human_judgments.json
```

The custom JSON is a list of 10–20 objects with `query_id` and `query`; optional `selection_category` is descriptive only. The default UI uses `results/human_evaluation/human_judgments.json`; review alternative study files separately or set `STUDY_PATH` to the intended study. Use a single judging process; concurrent writers and inter-reviewer agreement are not implemented. Back up real labels before switching configurations. After judging, run `summary`, rebuild the report and run verification again.

## Reproducible examples and architecture

`python -m src.submission_evidence` regenerates:

- `results/ranking_lab_example.json` and `.md`: real query `q_05836`; removing **Contract** changes the winner from K. Hymavathi to Sudhir Kumar Jain. The old winner moves to rank 4; top-five Jaccard is 4/6. This is ranking sensitivity, not legal importance or correctness.
- `results/failure_cases.json`: absent full Act phrase and unexpanded `sec. 302`, with actual empty outputs and alternate rankings. The app exposes both limitations.
- `results/grouping_diagnostic.json`: raw counts, missing/duplicate records, same-name multiple dates and name-normalization collisions.

`docs/architecture.dot` and report Figure 1 show the actual pipeline: parser and indexes → corpus-wide BM25/TF-IDF or derived Evidence-aware/RRF scores → literal/metadata eligibility → positive-score candidates → top-K → source windows → explanations. RRF is a separate BM25/TF-IDF branch, not downstream of Evidence-aware.

## Dataset credit and evaluation boundary

Source: Veningston K and Apratim Mishra, [IndicLegalQA Dataset, version 2](https://data.mendeley.com/datasets/gf8n8cnmvc/2), DOI 10.17632/gf8n8cnmvc.2, CC BY 4.0. The upstream description reports 10,000 QA pairs from 1,256 judgments. The supplied preprocessing groups by `(case_name, judgement_date)` and produces **1,260 local groups**; this local count must not be reported as an upstream judgment count. No new legal sources were crawled.

The diagnostic finds **1,253 cleaned names + seven extra date groups = 1,260 groups**, no missing names/dates/questions/answers, 14 duplicate complete QA records, and two normalized-name collision pairs. Group keys and QA counts agree with processed documents. Multiple dates can mean distinct decisions or metadata errors; without original judgment IDs/URLs, their relationship to the publisher's 1,256 judgments cannot be established. No groups were merged or dates rewritten.

The local documents concatenate answer passages by case. They are **not full judgments**, and the collection includes answers associated with evaluation queries. Thus the benchmark is case lookup over answer summaries, not independent evaluation on unseen judgments. Each query labels one case, so other relevant cases may be counted as nonrelevant. Categories are inferred from keywords. Literal anchor checks match section/article numbers only; they do not disambiguate the statute to which a number belongs.


## Planned continuation

Full judgment ingestion with source URLs and licenses; statute-aware anchors; manually judged pooled queries; semantic-index provenance checks; validated dense/cross-encoder comparisons; and uncertainty estimates on independent queries. Citation authority and precedent relationships are not implemented or invented.

## Interface finishing pass

The desk uses local CSS, a navy/teal and gold palette, responsive navigation, readable source cards, search-syntax help, and saved-study progress. Stopword-only queries return a useful empty state. Search caches refresh when the corpus changes. Statistical conclusions are generated from current results; undefined t-tests serialize as null instead of NaN.
