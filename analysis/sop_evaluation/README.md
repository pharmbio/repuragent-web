# SOP Retriever Evaluation

This folder contains a retrieval-only benchmark for the four SOP PDFs currently in the SOP-RAG corpus:

- `ANNEX3_R4ALL_SOP_DIPL-phospholipidosis.pdf`
- `REMEDI4All_SOPs_HCS_phospholipidosis_KQ_250210.pdf`
- `REMEDI4All_SOPs_In Silico_Repurposing.pdf`
- `SOP-INT-NA-1_3 Drug combination screening.pdf`

## Files

- `questions.csv`: annotated evaluation queries with gold source file, source page, and a short evidence snippet.
- `evaluate_retriever.py`: runs the current `SOPRetriever` and reports retrieval metrics.

## What The Script Measures

Because the current index stores `filename` metadata but not stable page or chunk identifiers, the benchmark reports two levels of retrieval quality:

- `file_hit`: whether the correct SOP file appears in the top-k retrieved items.
- `evidence_hit`: whether any retrieved item contains a gold evidence snippet from the benchmark.

The evidence-snippet match is a practical proxy for chunk-level retrieval until the indexer stores page numbers and chunk IDs.

## Run

From `/Users/dinhu955/Desktop/RepurAgent/repuragent-web`:

```bash
python analysis/sop_evaluation/evaluate_retriever.py
```

Optional arguments:

```bash
python analysis/sop_evaluation/evaluate_retriever.py --k 4
python analysis/sop_evaluation/evaluate_retriever.py --output-dir analysis/sop_evaluation/results_run_01
python analysis/sop_evaluation/evaluate_retriever.py --pdf-dir /Users/dinhu955/Desktop/RepurAgent/repuragent-web/persistence/data/SOP
python analysis/sop_evaluation/evaluate_retriever.py --skip-validate-gold
```

## Outputs

The script writes:

- `summary.json`
- `per_query_results.jsonl`

to a timestamped directory under `analysis/sop_evaluation/results/` unless `--output-dir` is provided.

## Notes

- The benchmark is intentionally retrieval-only. It does not score answer generation.
- Query wording includes both direct and paraphrased requests.
- The two phospholipidosis SOPs are treated as hard negatives for each other where appropriate.
