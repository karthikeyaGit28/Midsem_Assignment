# LegalLens Dataset Documentation

## Source
- **Benchmark:** IndicLegalQA Benchmark (Indian Supreme Court Judgments and Associated Legal Queries)
- **File:** `IndicLegalQA Dataset_10K_Revised.json`
- **Total Records:** 10,000 Q&A pairs
- **Unique Judgments:** 1,260 Supreme Court cases

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
  "text": "Full synthesized legal holding & factual narrative",
  "passages": ["Paragraph 1", "Paragraph 2", "..."],
  "num_qa_pairs": 8
}
```

### Processed Evaluation Splits
- `queries_all.json`: All 10,000 queries mapped to target `relevant_doc_id`
- `queries_val.json`: 1,000 validation queries used for hyperparameter / $\alpha$-tuning
- `queries_test.json`: 2,000 held-out test queries used for final IR evaluation metrics

## Preprocessing Pipeline
Run `python src/preprocessing.py` to regenerate processed datasets from raw files.
