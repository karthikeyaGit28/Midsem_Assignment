# Report claims mapped to executable checks

The final suite has 40 tests. Report claims describe these checks, rather than
asserting exhaustive correctness. Synthetic labels in tests are never saved in the
real study. `results/test_results.xml` records the executed suite.

- Original tokenization, inverted postings, DF/IDF, BM25 and TF-IDF: eight original tests in `tests/test_retrieval.py`.
- Aligned score arrays independent of JSON order: `test_score_alignment_is_independent_of_json_order`.
- Metadata applied before top-K: `test_filter_happens_before_top_k`.
- Literal stopwords and inflections: `test_literal_phrase_keeps_stopwords`, `test_literal_phrase_is_not_stemmed`.
- Quoted phrases cannot span separately supplied answers: `test_phrase_cannot_cross_two_original_answer_passages`.
- Literal exclusions: `test_exclusion_removes_source`.
- Exact number boundaries (30 versus 302, not statute identity): `test_section_number_does_not_match_prefix`.
- Raw BM25 contribution reconstruction with query repetition and title weighting: `test_term_audit_reconstructs_bm25`.
- Original excerpt substring, correct passage ID and evidence beyond word 120: `test_exact_evidence_is_a_source_substring`, `test_passage_after_word_120_is_retrievable`.
- No arbitrary zero-score results / empty query / nonpositive top-K: `test_no_arbitrary_zero_score_results`.
- Literal constraints retained in counterfactuals: `test_constraints_survive_counterfactuals`.
- Parser errors and minus signs inside quotes: `test_parse_errors_and_minus_in_quote`.
- Repeated RRF ranks and explicit document-ID tie order: `test_rank_fusion_is_reproducible`, `test_rank_fusion_ties_use_document_id_order`.
- Markdown evidence provenance: `test_evidence_brief_has_provenance`.
- Known-rank precision, recall, reciprocal rank and discounted gain; depth-ten cutoff: `test_metric_known_rank`, `test_metrics_ignore_relevance_beyond_depth_ten`.
- Escaped source HTML and stable highlight tags: `test_highlight_escapes_source_without_rewriting_its_own_tags`.

These retrieval checks are in `tests/test_research_engine.py` (19 tests).

`tests/test_human_evaluation.py` has 11 checks:

- No automatic relevance labels or numeric mean before completion.
- Disk reload and merge preserve labels for other queries.
- P@5 keeps denominator 5 even when fewer results are returned.
- Complete study arithmetic and pair counts.
- Clearing a label restores pending status.
- Partial top-five lists have no P@5 until all five are judged.
- Three parameterized checks reject unknown IDs, invalid labels and blank reviewers.
- Different study/corpus identifiers are rejected.
- Empty retrieval lists require explicit human review.
- Deterministic category sampling does not turn qrels into labels.

Some tests verify several related claims. The collected suite contains 11 cases
from this file, including the three parameterized validation cases.

`tests/test_app.py` has two Streamlit integrations:

- `test_search_and_judgment_persistence`: submit queries, restore exploratory labels,
  handle no-match results and malformed quotes.
- `test_formal_human_labels_survive_new_app_session`: save labels to a temporary
  study, start a new app instance, restore labels from disk, retain pending mean.

Full-corpus verification is additional to unit tests:

- `python -m src.evaluate_research`: 2,000 supplied queries, four lexical methods,
  80 repeated rankings for the first 20 queries, complete per-query artifact.
- `python -m src.submission_evidence`: reruns original/modified worked example;
  captures actual failure outputs and verifies processed grouping keys/QA counts.
- `python -m src.verify_submission`: checks hashes, counts, rankings-derived metric
  aggregates, CSV/JSON agreement, study status, passing XML and report fingerprints.
- Browser inspection: a real `"rent control act"` query returned four cases; the
  formal study displayed 0/15 queries and 0/75 pairs complete. No real label was set.
- PDF QA: eight pages rendered and visually inspected; report-builder page limit
  and artifact hashes are checked separately.

Not established by these checks: legal correctness, case authority, full-judgment
retrieval quality, inter-reviewer agreement, statistical significance, dense model
performance, or identical results across arbitrary dependency versions.
