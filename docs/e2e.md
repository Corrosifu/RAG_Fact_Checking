# End-to-end evaluation

The unit tests check the code; the SciFact benchmark checks claim verification
([`evaluation.md`](evaluation.md)). This page is about the third thing: does the **whole
pipeline** answer questions correctly, decline the ones it cannot answer, and verify claims?
`ragfc e2e` runs it on a golden set of 30 items and scores every stage **without an LLM judge**
(free and reproducible).

## Run it

```bash
ragfc e2e                                   # lightweight components, deterministic
ragfc e2e --baseline tests/e2e/baseline.json   # also check against the reference scores (= make e2e)
RAGFC_EMBEDDER=sentence-transformers RAGFC_VERIFIER=nli \
  ragfc e2e --verifier env --output docs/results/e2e-golden-minilm-nli-base.json   # real models
```

`--update-baseline` records new reference scores (needs `--baseline FILE`); `--tolerance` changes
the allowed drop (default 0.05).

## The golden set (`src/rag_fact_checking/sample/golden/`)

25 short documents written for this project (no licence issue) and 30 items:

| Kind | Items | What it tests |
|---|---|---|
| `factual` | 8 | one document answers the question |
| `multi_doc` | 3 | the answer needs two or three documents |
| `yes_no` | 3 | polarity; includes a near miss (BERT vs the decoder-only GPT) |
| `numeric` | 3 | a number must be found |
| `unanswerable` | 4 | the corpus does not contain the answer; one is only answerable from a **reference list**, which the pipeline drops |
| `claim` | 9 | 3 supported, 3 refuted, 3 not enough info |

Traps are deliberate: *quantization* also exists as a chemistry document, a document ends with a
bibliography, and several documents share vocabulary. **The references and evidence sentences are
my own; they deserve your review.** The data lives in the package (`corpus.jsonl`,
`questions.jsonl`); `tests/test_e2e.py` checks that every evidence sentence is quoted word for
word from a gold document.

## Metrics

| Metric | Definition |
|---|---|
| `retrieval.hit` | a gold document is among the retrieved chunks |
| `retrieval.context_recall` | share of gold evidence sentences found word for word in a retrieved chunk |
| `retrieval.context_precision` | share of retrieved chunks that come from a gold document |
| `answer.f1`, `answer.contains` | token F1 against the best reference; a reference appears in the answer |
| `answer.relevancy` | cosine similarity between the question and the answer |
| `answer.yes_no_*` | parse rate and accuracy when the answer starts with yes or no (generative models) |
| `faithfulness` | share of answer sentences the verifier confirms |
| `abstention.rate` | on unanswerable questions: the system declines (fallback sentence, or no sentence confirmed) |
| `abstention.false_rate` | the same on answerable questions: lower is better |
| `claims.accuracy` | SUPPORTED / REFUTED / NOT_ENOUGH_INFO verdict accuracy |
| `latency.*` | mean, median and 95th percentile per stage (retrieve, generate, verify) |

The metric names follow the usual RAG evaluation vocabulary (context recall and precision,
faithfulness, answer relevancy) but are computed without a judge model, so they are **not the
RAGAS numbers**. An optional RAGAS run is planned in the next step.

## Two tiers

- **Tier A (CI).** Hashing embedder, extractive generator and `OverlapVerifier`, a deterministic
  stand-in for the NLI model. `tests/test_e2e.py` runs it on every PR and fails if a gated metric
  (`retrieval.hit`, `retrieval.context_recall`, `answer.f1`, `answer.contains`, `faithfulness`,
  `abstention.rate`, `claims.accuracy`) drops by more than 0.05 below `tests/e2e/baseline.json`.
  Two tests break a component on purpose to prove the gate notices. It protects the plumbing
  (chunking, retrieval, evidence windows, aggregation) and says **nothing about the quality of
  a real NLI model**.
- **Tier B (manual).** The same set with real models, informative only.

To accept a deliberate change of the scores, run `ragfc e2e --baseline tests/e2e/baseline.json
--update-baseline` and let the reviewer see the diff of `baseline.json`. A baseline recorded
with a different setup (embedder, generator, chunking) is refused.

## Results

Golden set, extractive generator, `all-MiniLM-L6-v2` retrieval, 4 CPU cores
(`docs/results/e2e-golden-*.json`). Thirty items: differences of a few points mean nothing.

| | Tier A (hashing + double) | MiniLM + base NLI | MiniLM + fine-tuned NLI (v2) |
|---|---|---|---|
| retrieval.hit | 0.96 | 1.00 | 1.00 |
| retrieval.context_recall | 0.93 | 1.00 | 1.00 |
| answer.f1 / answer.contains | 0.33 / 0.71 | 0.38 / 0.71 | 0.38 / 0.71 |
| faithfulness | 1.00 | **0.27** | 0.83 |
| abstention.rate (unanswerable) | 0.00 | 0.50 | 0.00 |
| abstention.false_rate | 0.00 | 0.65 | 0.18 |
| claims.accuracy | 1.00 | 0.78 | 1.00 |
| verify latency per answer | under 1 ms | 676 ms | 609 ms |

What it shows:

- **The extractive generator never declines.** All four unanswerable questions get an answer (the
  best-matching sentence), and it sometimes quotes the wrong sentence: asked *why cross-encoders are
  applied only to the top candidates*, it returns a neighbouring sentence. Its token F1 is about 0.35.
- **The base NLI model is a poor answer checker.** It confirms only 27 % of the sentences of an
  answer that quotes the source word for word, so it would flag most good answers
  (a false abstention rate of 0.65). The model fine-tuned on SciFact confirms 83 %.
  The "abstention" of the base model is therefore mostly a false alarm, not a real refusal.
- **Retrieval is solved on this easy set** (hit 1.00 with MiniLM, 0.96 with hashing); the
  difficulty is in the generator and in the verifier.

## Limits

30 items written by one person and not yet reviewed; short documents; no LLM-generated or
free-form reference answers yet (token F1 punishes correct paraphrases); `answer.relevancy`
is a cheap embedding similarity; the tier A baseline measures a test double, not a model.
The larger QASPER run and the optional RAGAS adapter come in the next step.

## A bug this found

The first runs showed an extractive answer that lost its subject
(*"builds a subword vocabulary..."* instead of *"Byte pair encoding builds..."*): `strip_title`
removed the document title wherever it appeared, including when it began a sentence. It now keeps
a title directly followed by a lower-case word (a page header is followed by a capital or a new
line). A regression test covers it.
