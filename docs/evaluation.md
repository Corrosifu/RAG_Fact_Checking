# Evaluation on SciFact

Everything here is measured on the SciFact **validation** split (300 claims, the original dev
set; the public test split has no labels), with a single deterministic run. The raw results are
in [`results/`](results/).

## 1. Reproduce

```bash
pip install -e ".[ml,bench]"
export RAGFC_VERIFIER=nli RAGFC_EMBEDDER=sentence-transformers

# baseline: off-the-shelf NLI model
ragfc eval-scifact --output docs/results/scifact-validation-minilm.json

# fine-tune on SciFact train (CPU, several minutes per epoch), then evaluate the checkpoint
ragfc finetune-nli --hard-negatives 3 --epochs 2 --out models/nli-scifact --output docs/results/finetune-nli-v2-hard-negatives.json
ragfc eval-scifact --threshold 0.7 --verifier-model models/nli-scifact/best
```

The first run downloads the data (Hugging Face parquet conversion of `allenai/scifact`) and the
models, then indexes 5,183 abstracts; the index is cached. `--limit N` runs the first N claims,
`--no-verify` runs retrieval only, `--extra-corpus nfcorpus scidocs` adds unrelated abstracts.
The fine-tuned weights are not committed (330 MB).

## 2. Setup

| | |
|---|---|
| Data | 300 claims, 5,183 abstracts; labels SUPPORT -> SUPPORTED, CONTRADICT -> REFUTED, cited abstracts without evidence -> NOT_ENOUGH_INFO (124 / 64 / 112 claims) |
| Chunking | 200-word windows, overlap 40, reference removal off (abstracts have no bibliography) |
| Retrieval | hybrid BM25 + dense (`all-MiniLM-L6-v2`, 22.7 M parameters) fused with RRF |
| Verifier | `cross-encoder/nli-MiniLM2-L6-H768` (82.1 M parameters), ~60-word evidence windows chosen by `select_premises`, same `verify_claim` code as `RAGPipeline.fact_check` |
| Machine | 4 CPU cores, no GPU |

Metrics:

- **Retrieval**: hit@k, share of the 188 claims with evidence that have a gold abstract in the top k.
- **Oracle**: only the cited abstracts are available (isolates the verifier). 3-class accuracy and macro-F1; always answering the majority class gives 41.3 %.
- **End to end**: retrieve from the whole corpus, then verify. Same metrics.
- **Abstract Label-Only F1**: the metric published SciFact systems report. A predicted (claim, abstract, label) is correct if the abstract is a gold evidence abstract and the label matches; `NOT_ENOUGH_INFO` is not a prediction; micro precision / recall / F1 (definition from Wadden et al., 2020). "Open" judges the top 3 retrieved abstracts (the cut-off VeriSci used); "abstracts given" judges the cited ones. **This is my implementation of the definition, not the official scoring script.**

## 3. Fine-tuning the NLI verifier

Training data comes from the 809 training claims, split 687 / 122 by claim (stratified, seed 0).
The 122 held-out claims choose the epoch and the decision threshold; the validation split is used
only for the final numbers. Premises are built with the product's own `select_premises`, so the
model trains on the windows it sees at inference; a window containing a gold rationale sentence
is labelled entailment (SUPPORT) or contradiction (CONTRADICT), every other window neutral.

| | v1 | v2 (hard negatives) |
|---|---|---|
| Training pairs (entailment / contradiction / neutral) | 531 / 290 / 1,160 | 531 / 290 / 3,011 |
| Extra neutral windows | none | up to 3 per claim from retrieved abstracts that are not evidence |
| Epochs run, best epoch (on hold-out) | 3, epoch 1 | 2, epoch 1 |
| Selection score on hold-out | oracle macro-F1 | open top-3 Abstract Label-Only F1 |
| Hyper-parameters | AdamW, lr 2e-5, batch 16, max length 192, class-weighted cross-entropy, seed 0 | same |

Hard negatives were added because v1 showed, on validation, that the open-setting precision was
low: the verifier labelled look-alike abstracts that are not evidence. MultiVerS reports the same
effect and trains with 20 sampled negative abstracts per claim. **v2 was therefore designed after
seeing v1's validation results**; its epoch and threshold were still chosen on the hold-out only.

### Results (validation, MiniLM retrieval)

| Verifier | Oracle accuracy | Oracle macro-F1 | End-to-end accuracy | End-to-end macro-F1 |
|---|---|---|---|---|
| Base NLI, threshold 0.6 | 54.7 % | 0.514 | 47.7 % | 0.453 |
| Fine-tuned v1, threshold 0.6 | 65.7 % | 0.644 | 55.0 % | 0.551 |
| Fine-tuned v2, threshold 0.7 | 64.3 % | 0.617 | 60.0 % | 0.588 |

| Verifier | Abstract Label-Only F1, abstracts given | F1, open (top 3) | Precision / recall, open (top 3) |
|---|---|---|---|
| Base NLI | 0.437 | 0.297 | 0.319 / 0.278 |
| Fine-tuned v1 | 0.608 | 0.362 | 0.302 / 0.450 |
| Fine-tuned v2 | 0.572 | 0.411 | 0.420 / 0.402 |

What it shows:

- Fine-tuning on 687 claims lifts end-to-end accuracy from 48 % to 60 % (about 19 points above the 41 % majority baseline).
- v1 is the better verifier when the right abstracts are given; v2 is better in the open setting, where it matters, because it learned to leave look-alike abstracts alone (precision 0.30 -> 0.42).
- Per class (v2, oracle): when it says `SUPPORTED` it is right 59 times out of 59, but it still finds only 48 % of the supported claims (recall 0.476); `REFUTED` precision 0.60, recall 0.45.
- Tuning the **base** model's threshold on the hold-out did not rescue it: threshold 0.3 raised its oracle accuracy to 58.7 % but dropped end-to-end accuracy to 41.3 % ([`results/scifact-validation-base-threshold0.3.json`](results/scifact-validation-base-threshold0.3.json)).
- Differences of a few points between v1 and v2 are inside the noise of 300 claims (about +/-6 points) and of a single seed; the 12-point gain over the base model is not.

## 4. Comparison with published systems

Abstract Label-Only F1 (higher is better). Published numbers are copied from the papers; mine are
on the validation split with the caveats below.

| System | Parameters | Open: retrieve, then label | Abstracts given |
|---|---|---|---|
| **This project, base NLI** | 82 M + 23 M embedder | 29.7 | 43.7 |
| **This project, fine-tuned v2** | same | **41.1** | 57.2 |
| This project, fine-tuned v1 | same | 36.2 | 60.8 |
| VeriSci, zero-shot (trained on FEVER), dev | two RoBERTa-large | 46.5 | 70.4 |
| VeriSci, dev | two RoBERTa-large + TF-IDF top 3 | 51.0 | 77.4 |
| VeriSci, test | same | 47.4 | 74.7 |
| ParagraphJoint | 360 M | 69.1 | 74.1 |
| VerT5erini | 5.6 B | 68.2 | 81.7 |
| MultiVerS | 440 M | 72.5 | 80.9 |
| Human agreement | | n/a | 89.1 |

Sources: Wadden et al. 2020, *Fact or Fiction* (arXiv 2004.14974), tables 4 and 7 (VeriSci, with
a bootstrap standard deviation of about 3 F1); Wadden et al. 2022, *MultiVerS* (arXiv 2112.01640),
tables 2 and 5 (the other systems and the human figure; the passages I read do not say which
split these tables use).

Reading it honestly:

- **The light system does not compete with the state of the art.** In the open setting it reaches 41 F1 against 68-73 for recent systems, and 57-61 against 74-82 when the abstracts are given. It also stays below the first published baseline (51 dev, 47 test).
- **It is in the same range as that first baseline**, with a fraction of its size (VeriSci uses two RoBERTa-large models) and no GPU: 0.3 s per claim on 4 CPU cores. It uses about 4 times fewer parameters than MultiVerS and 50 times fewer than VerT5erini (105 M against 440 M and 5.6 B), and reaches 57 % of MultiVerS's open-setting F1 (41.1 / 72.5).
- **Where the gap probably comes from** (hypotheses, not tested here): no biomedical pretraining and a small encoder; 687 training claims and 1-2 epochs against 20 epochs, FEVER and S2ORC pretraining and 20 negatives per claim for MultiVerS; 60-word windows instead of the full abstract (MultiVerS reads whole abstracts with a Longformer); a weaker retriever than VerT5erini's BM25 + neural re-ranker.
- Recall is the weak side everywhere: the verifier prefers `NOT_ENOUGH_INFO` to a guess. That is a deliberate property of a fact-checker, and it is what the F1 here penalises.

Why the numbers are not strictly comparable: my metric implementation, my split (validation, not
the hidden test set), my retriever and cut-offs (top 3), a single run, 300 claims. A fair
comparison would submit predictions to the SciFact leaderboard, which I could not do.

## 5. Does a bigger corpus help?

SciFact's corpus already contains the gold evidence of every claim, so enlarging it cannot help
this benchmark; it can only add distractors. The experiment measures what that costs, using the
v2 verifier and unrelated abstracts from BEIR collections (NFCorpus: medicine and nutrition,
SCIDOCS: scientific papers across fields).

| Corpus | Documents | hit@5 | MRR | End-to-end accuracy | Abstract Label-Only F1, open (top 3) | Seconds per claim |
|---|---|---|---|---|---|---|
| SciFact | 5,183 | 0.904 | 0.802 | 60.0 % | 0.411 | 0.32 |
| + NFCorpus | 8,816 | 0.872 | 0.779 | 61.0 % | 0.415 | 0.33 |
| + NFCorpus + SCIDOCS | 34,129 | 0.872 | 0.783 | 59.7 % | 0.422 | 0.39 |

- Retrieval loses about 3 points of hit@5 once the corpus grows, then holds up to 6.6 times the size.
- The verification results do not move (differences of 1 point are noise). The verifier filters out the irrelevant retrieved abstracts, so extra documents mostly cost time.
- The distractors are not adversarial (they were not chosen to resemble SciFact claims), so this is a favourable test; a corpus full of near-duplicates of the evidence would hurt more.

What it means for the arXiv use case: the corpus has to **contain the evidence**; size on its
own neither helps nor hurts much at this scale. Enrich it by topic (the fields you will check
claims about, full text where abstracts do not carry the finding) rather than in bulk. An unseen
claim over a corpus that lacks its evidence correctly ends in `NOT_ENOUGH_INFO`, not in a wrong
verdict. At the sizes tested the exact numpy index is sufficient; the optional FAISS backend
becomes useful above roughly 50,000 chunks.

## 6. Limits of this evaluation

- 300 claims: the 95 % margin on an accuracy near 50-60 % is about +/-6 points. Single seed, no confidence intervals computed.
- Epoch and threshold were chosen on only 122 held-out claims, so they are noisy; the held-out score of the selected epoch is optimistic by construction.
- v2 was designed after looking at v1's validation behaviour (see section 3). I looked at several configurations on validation (base at two thresholds, v1, v2, and the hashing embedder) plus the corpus experiment; none was selected on validation.
- Labels are per claim and per abstract; sentence-level rationales are not scored.
- Abstract Label-Only F1 is my implementation of the published definition, not the official script, and the comparison with published systems mixes splits and retrieval set-ups.
- The breakdown of supported claims in the earlier base-model analysis comes from a one-off script, not from the CLI. The CLI commands themselves have no end-to-end automated test beyond their building blocks.
