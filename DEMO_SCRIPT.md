# LegalLens live demo - approximately 7 minutes

Show the running app and actual code. No slides are needed. Adapt the speaker roles below to the team's actual work.

## 0:00-0:45 - Problem and track (speaker 1)

Open the app at http://127.0.0.1:8501. Say: "LegalLens is our Track T6 retrieval prototype. The research question is not just which case comes first, but whether we can inspect and challenge why it comes first. Our local collection consists of case-linked answer passages from IndicLegalQA, not full judgment texts."

## 0:45-1:35 - Corpus and IR foundations (speaker 1)

Open `data/processed/documents.json`, then `src/indexing.py` and `src/bm25_retriever.py`. Show term frequencies, positions, document frequency, IDF, and the case-title zone boost. Explain stemming for lexical retrieval versus literal tokens for exact phrases.

## 1:35-2:40 - Real source retrieval (speaker 2)

Click **Tenancy & notice**. Show the returned cases and real passage numbers. Expand **Inspect evidence & score** on the first source. Explain that the contributions reconstruct its raw BM25 score; the final evidence-aware score uses normalized BM25 and a disclosed anchor bonus. The example may return an imperfect first result: show the excerpt and acknowledge the limitation.

## 2:40-3:35 - Domain structure (speaker 2)

Click **Exact source phrase** (`"rent control act"`) to show a positive literal match. Search `"section 302" IPC`, select **Require every section / article anchor**, and submit. Explain rare-term-first intersection and within-passage adjacency. In **Benchmarks**, expand the saved failure demonstrations: `"sec. 302"` returns no results while `"section 302"` returns five. No abbreviation normalization or statute disambiguation is claimed.

## 3:35-4:55 - Ranking Lab, the main novelty (speaker 3)

Click **Tenancy & notice**, submit it if needed, and open **Ranking Lab**. Show the BM25 contribution chart, parsed query, and method comparison. Click **Run leave-one-term-out experiment**. Read one actual changed winner or reduced overlap from the computed table; do not promise a particular word always changes the winner. Explain: "This is a controlled rerun with one term removed, so we can test our explanation rather than accepting a static score badge." Download the evidence brief or counterfactual JSON if useful.

Expand **Saved reproducible worked example from the dataset**. Query `q_05836` asks about Section 25(3) of the Indian Contract Act. Removing **Contract** changes the winner from K. Hymavathi to Sudhir Kumar Jain; the original winner moves to rank 4, and top-five overlap is 4/6. It demonstrates sensitivity, not legal importance or correctness.

## 4:55-6:05 - Evaluation and judging (speaker 4)

Open **Benchmarks → A. Dataset-qrel evaluation**. Show the reproduced 2,000-query results and hashes. BM25 MRR@10 is 0.771174, Evidence-aware 0.771465, and RRF 0.762292. The gain is tiny and not significance-tested; RRF loses here. Open **Judge relevance → B. Human-judged evaluation**. Show 15 frozen queries and enter a real reviewer name/initials. Save actual labels for one query, reload to demonstrate disk persistence, and export. P@5 uses denominator 5; the study mean stays pending until all 75 pairs are judged. If judging is incomplete, show that status explicitly.

## 6:05-7:00 - Honest limitations and continuation (speaker 4)

Show **Inside the pipeline** and say: "The novelty is an integrated inspection workflow, not a newly invented BM25 or RRF algorithm. Our corpus contains answer passages linked to evaluation queries, so this is a summary-level case-lookup benchmark. One labelled case per query is incomplete relevance information; keyword categories can be wrong. We plan to ingest full judgments, disambiguate statutes, and collect a manually judged pool."

Finish by showing the recorded 40-test passing suite, the verification artifact, actual team contributions and Codex assistance. Explain the grouping diagnostic: 1,253 cleaned names plus seven extra date groups; upstream identity remains unresolved. Use `REPORT_ENHANCED.pdf` and replace all four placeholders with verified names/work. The original historical report is not a description of this revised app.
