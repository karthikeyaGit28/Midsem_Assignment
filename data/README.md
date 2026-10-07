# LegalLens Dataset Documentation

The enhanced app retrieves **case-linked answer passages**, not full judgment texts.
The local representation concatenates the dataset answers; it includes answers linked
to validation and test questions. Metrics describe case lookup over answer summaries,
not independent full-judgment retrieval. Categories are keyword-based heuristics.

Credit: Veningston K and Apratim Mishra (2024),
[IndicLegalQA Dataset, version 2](https://data.mendeley.com/datasets/gf8n8cnmvc/2),
DOI 10.17632/gf8n8cnmvc.2, CC BY 4.0. The publisher describes **1,256 judgments**;
the supplied `(case_name, judgement_date)` grouping produces **1,260 local groups**.

## Source
- **Benchmark:** IndicLegalQA Benchmark (Indian Supreme Court Judgments and Associated Legal Queries)
- **File:** `data/raw/IndicLegalQA_Dataset_10K_Revised.json`
- **Total Records:** 10,000 Q&A pairs
- **Local Case Groups:** 1,260 groups by case name and date

## Grouping diagnostic

Run `python -m src.submission_evidence`. `results/grouping_diagnostic.json` records
1,253 cleaned names, seven same-name extra dates, 1,260 groups, zero missing fields,
14 duplicate complete QA records and two punctuation/case-normalized name collision
pairs. Processed keys and QA counts match the existing grouping rule. The seven date
pairs may reflect distinct decisions or metadata errors; source judgment IDs and URLs
are absent, so the upstream 1,256-judgment correspondence remains unresolved. The
diagnostic reports these findings without changing dates, groups or document IDs.

## Schema
### Raw Record
```json
{
  "case_name": "Union of India vs. Maj. Gen. Manomoy Ganguly",
  "judgement_date": "1st August 2018",
  "question": "Who is the respondent in the case Union of India vs. Maj. Gen. Manomoy Ganguly?",
  "answer": "The respondent is Maj. Gen. Manomoy Ganguly."
}
```

### Processed Documents (`data/processed/documents.json`)
```json
{
  "doc_id": "doc_0000",
  "case_name": "String",
  "date": "String",
  "court": "Supreme Court of India",
  "category": "Civil / Criminal / Constitutional / Service / Family / etc.",
  "text": "Case metadata followed by concatenated source answer passages",
  "passages": ["Paragraph 1", "Paragraph 2", "..."],
  "num_qa_pairs": 8
}
```

### Processed Evaluation Splits
- `queries_all.json`: All 10,000 queries mapped to target `relevant_doc_id`
- `queries_val.json`: 1,000 validation queries; the original dense project used them for alpha tuning
- `queries_test.json`: 2,000 test questions; enhanced sparse settings are fixed, not tuned here

These are query-level splits. All source answers were indexed before splitting, so
this is not an independent unseen-judgment benchmark. Human study labels are stored
separately in `results/human_evaluation/` and are never inferred from dataset qrels.

## Preprocessing Pipeline
Run `python src/preprocessing.py` to regenerate processed datasets from raw files.
