# Evaluation on SciFact

## Reproduce

```bash
pip install -e ".[ml,bench]"
export RAGFC_VERIFIER=nli
RAGFC_EMBEDDER=sentence-transformers ragfc eval-scifact --output docs/results/scifact-validation-minilm.json
RAGFC_EMBEDDER=hashing               ragfc eval-scifact --output docs/results/scifact-validation-hashing.json
```

The first run downloads the data (Hugging Face parquet conversion of `allenai/scifact`) and the
models, then indexes 5,183 abstracts (a few minutes with MiniLM on CPU); the index is cached.
`--limit N` runs the first N claims as a smoke test, `--no-verify` runs retrieval only.

## Setup

| | |
|---|---|
| Data | SciFact `validation` split (the original dev set): 300 claims, 5,183 abstracts. The public test split has no labels |
| Labels | `SUPPORT` -> SUPPORTED, `CONTRADICT` -> REFUTED, cited abstracts without evidence -> NOT_ENOUGH_INFO (124 / 64 / 112 claims) |
| Chunking | 200-word windows, overlap 40, reference removal off (abstracts have no bibliography) |
| Retrieval | hybrid BM25 + dense, fused with RRF; top 5 chunks for verification |
| Verifier | `cross-encoder/nli-MiniLM2-L6-H768`, not fine-tuned on SciFact; threshold 0.6 fixed in advance |
| Code path | the benchmark calls the same `verify_claim` as `RAGPipeline.fact_check` |

Three experiments, each isolating one component:

1. **Retrieval**, on the 188 claims that have gold evidence: is a gold abstract among the top documents?
2. **Oracle verification**: only the abstracts the annotators cited are available, so the NLI model and the premise selection are measured alone.
3. **End to end**: retrieve from the whole corpus, then verify.

## Results

### Retrieval (188 claims with evidence, abstract level)

| Embedder | hit@1 | hit@5 | hit@10 | recall@5 | MRR |
|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 0.713 | 0.904 | 0.952 | 0.886 | 0.802 |
| `hashing` (default, no model) | 0.505 | 0.750 | 0.814 | 0.731 | 0.619 |

`hit@k` is the share of claims with at least one gold abstract in the top k documents;
`recall@k` averages the share of gold abstracts found.

### Verification (300 claims, 3 classes; majority-class baseline 0.413)

| Experiment | Accuracy | Macro-F1 | Seconds per claim (CPU) |
|---|---|---|---|
| Oracle (cited abstracts given) | 0.547 | 0.514 | 0.07 |
| End to end, MiniLM retrieval | 0.477 | 0.453 | 0.28 |
| End to end, hashing retrieval | 0.460 | 0.438 | 0.28 |

The oracle result is identical for both embedders, as it must be: no retrieval is involved.

Per class (oracle):

| Class | Precision | Recall | F1 | Claims |
|---|---|---|---|---|
| SUPPORTED | 0.917 | 0.266 | 0.413 | 124 |
| REFUTED | 0.558 | 0.453 | 0.500 | 64 |
| NOT_ENOUGH_INFO | 0.481 | 0.911 | 0.630 | 112 |

Confusion matrix (oracle; rows = gold, columns = predicted):

| | SUPPORTED | REFUTED | NOT_ENOUGH_INFO |
|---|---|---|---|
| SUPPORTED | 33 | 13 | 78 |
| REFUTED | 3 | 29 | 32 |
| NOT_ENOUGH_INFO | 0 | 10 | 102 |

## Reading the results

- **The verifier, not retrieval, limits the system.** It rarely says `SUPPORTED` wrongly (33 of 36
  predictions are right) but misses most supported claims. A diagnosis on the 124 supported claims:
  5 had no evidence window selected, 73 had windows but the NLI model found no entailment
  (median top entailment probability 0.10), 13 were called refuted, 33 were right.
- That points at the model rather than the window selection: SciFact claims paraphrase
  biomedical findings, while this NLI model was trained on general-purpose entailment data, and
  the premise selection was tuned by hand on a few computer-science papers (domain shift).
- Refusing to guess (`NOT_ENOUGH_INFO` when evidence is weak or conflicting) explains the high
  precision and low recall. It is the safer failure mode, but it caps accuracy here.
- End to end, the system is about 6 points above always answering the majority class.

## Limits of this evaluation

- 300 claims: the 95 % margin on an accuracy near 50 % is about ±6 points.
- One deterministic run, no repeated seeds, no confidence intervals computed.
- Labels are per claim and per abstract. Sentence-level rationales are not scored, so the
  "exact evidence sentences" feature is not evaluated here.
- No comparison with published SciFact systems: their setups and metrics differ from this
  claim-level, 3-class protocol, so a side-by-side would mislead.
- The breakdown of the 124 supported claims above comes from a one-off diagnostic script, not
  from `ragfc eval-scifact`; the CLI itself has no automated test beyond its building blocks.
- The threshold (0.6) was not tuned. Tuning it on the training split and then reporting on
  validation is a planned improvement, as is fine-tuning the NLI model on the training claims.
