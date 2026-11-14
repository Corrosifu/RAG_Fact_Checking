"""Command line interface: ``ragfc <command>``."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from importlib import resources

from rag_fact_checking.config import Settings


def _sample_file(name: str):
    return resources.files("rag_fact_checking").joinpath("sample", name)


def cmd_sample(args: argparse.Namespace, settings: Settings) -> int:
    settings.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with resources.as_file(_sample_file("corpus.jsonl")) as src:
        shutil.copyfile(src, settings.corpus_path)
    print(f"Sample corpus written to {settings.corpus_path}. Next: ragfc index")
    return 0


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    from rag_fact_checking.corpus import save_documents
    from rag_fact_checking.ingestion import ingest

    docs = ingest(args.topic, args.max_results, settings.pdf_dir, full_text=args.full_text)
    n = save_documents(docs, settings.corpus_path)
    print(f"Saved {n} documents to {settings.corpus_path}. Next: ragfc index")
    return 0


def cmd_index(args: argparse.Namespace, settings: Settings) -> int:
    from rag_fact_checking.corpus import build_index

    kb = build_index(settings)
    print(f"Indexed {len(kb)} chunks with {kb.embedder_name} into {settings.index_dir}")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    from rag_fact_checking.pipeline import RAGPipeline

    answer = RAGPipeline.from_settings(settings).ask(args.question, args.top_k)
    print(answer.text)
    if answer.checks:
        marks = {"supported": "✓", "contradicted": "✗", "unsupported": "?"}
        n_ok = sum(c.status == "supported" for c in answer.checks)
        print(f"\nVerification: {n_ok}/{len(answer.checks)} sentences supported by the sources")
        for check in answer.checks:
            print(f"  {marks[check.status]} {check.sentence}")
    print("\nSources:")
    for i, s in enumerate(answer.sources, start=1):
        print(f"  [{i}] {s.title or s.doc_id} ({s.chunk_id}, score={s.score})")
    return 0


def cmd_check(args: argparse.Namespace, settings: Settings) -> int:
    from rag_fact_checking.pipeline import RAGPipeline

    result = RAGPipeline.from_settings(settings).fact_check(args.claim, args.top_k)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from rag_fact_checking.embeddings import build_embedder
    from rag_fact_checking.evaluation.metrics import evaluate_retrieval, load_testset
    from rag_fact_checking.retriever import HybridRetriever

    path = args.testset
    if path is None:
        with resources.as_file(_sample_file("testset.jsonl")) as p:
            testset = load_testset(p)
    else:
        testset = load_testset(path)
    retriever = HybridRetriever.from_settings(settings, build_embedder(settings))
    print(json.dumps(evaluate_retrieval(retriever, testset, args.k), indent=2))
    return 0


def _scifact_retriever(settings, root, documents, extra, extra_names):
    """Index (or load the cached index of) SciFact + optional distractors; return the retriever."""
    import dataclasses
    import re

    from rag_fact_checking.corpus import build_index, save_documents
    from rag_fact_checking.embeddings import build_embedder
    from rag_fact_checking.index import KnowledgeBase, KnowledgeBaseNotFound
    from rag_fact_checking.retriever import HybridRetriever

    embedder = build_embedder(settings)
    # Abstracts have no bibliography, so the reference heuristics are switched off.
    run = dataclasses.replace(
        settings,
        data_dir=root
        / (
            "run-"
            + re.sub(r"[^A-Za-z0-9._-]+", "_", embedder.name)
            + "".join(f"+{n}" for n in extra_names)
        ),
        drop_references=False,
    )
    try:
        KnowledgeBase.load(run.index_dir, expected_embedder=embedder.name)
    except (KnowledgeBaseNotFound, ValueError):
        print(
            f"Indexing {len(documents) + len(extra)} documents with {embedder.name} ...",
            file=sys.stderr,
        )
        save_documents([*documents, *extra], run.corpus_path)
        build_index(run, embedder)
    return embedder, run, HybridRetriever.from_settings(run, embedder)


def cmd_eval_scifact(args: argparse.Namespace, settings: Settings) -> int:
    import dataclasses
    from datetime import date
    from pathlib import Path

    from rag_fact_checking.evaluation import scifact
    from rag_fact_checking.verifier import build_verifier

    root = Path(args.data_dir) if args.data_dir else settings.data_dir / "scifact"
    documents = scifact.build_documents(scifact.read_parquet(scifact.download(root, "corpus")))
    rows = scifact.read_parquet(scifact.download(root, args.split))
    extra = [
        d
        for name in args.extra_corpus
        for d in scifact.build_extra_documents(
            scifact.read_parquet(scifact.download_extra(root, name)), name
        )
    ]
    claims = scifact.build_claims(rows)
    if args.limit:
        claims = claims[: args.limit]

    embedder, run, retriever = _scifact_retriever(
        settings, root, documents, extra, args.extra_corpus
    )

    if args.verifier_model:
        settings = dataclasses.replace(settings, verifier_model=args.verifier_model)
    verifier = None if args.no_verify else build_verifier(settings)
    if verifier is None:
        print("No verifier: only the retrieval experiment runs.", file=sys.stderr)
    result = scifact.run_scifact(
        retriever,
        verifier,
        claims,
        documents,
        chunk_size=run.chunk_size,
        chunk_overlap=run.chunk_overlap,
        top_k=settings.top_k,
        threshold=args.threshold,
        progress=lambda msg: print(msg, file=sys.stderr),
    )
    result = {
        "benchmark": "SciFact",
        "split": args.split,
        "date": date.today().isoformat(),
        "setup": {
            "embedder": embedder.name,
            "verifier": settings.verifier_model if verifier else None,
            "chunk_size": run.chunk_size,
            "chunk_overlap": run.chunk_overlap,
            "top_k": settings.top_k,
            "nli_threshold": args.threshold,
            "documents": len(documents) + len(extra),
            "extra_corpus": list(args.extra_corpus),
        },
        **result,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_finetune_nli(args: argparse.Namespace, settings: Settings) -> int:
    from pathlib import Path

    from rag_fact_checking.evaluation import scifact
    from rag_fact_checking.training.finetune import finetune
    from rag_fact_checking.training.pairs import build_pairs, label_counts, split_claims
    from rag_fact_checking.verifier import NLIVerifier

    root = Path(args.data_dir) if args.data_dir else settings.data_dir / "scifact"
    corpus_rows = scifact.read_parquet(scifact.download(root, "corpus"))
    documents = scifact.build_documents(corpus_rows)
    sentences = {str(r["doc_id"]): list(r["abstract"]) for r in corpus_rows}
    rows = scifact.read_parquet(scifact.download(root, "train"))
    claims, rationales = scifact.build_claims(rows), scifact.build_rationales(rows)
    train_claims, held = split_claims(claims, args.holdout, args.seed)
    retriever = hard_negatives = None
    if args.hard_negatives:
        # Retrieved abstracts that are not evidence: the confusable documents the verifier
        # must learn to leave alone in the open setting.
        _, _, retriever = _scifact_retriever(settings, root, documents, [], [])
        hard_negatives = {
            c.id: [
                d
                for d in scifact.unique_docs(retriever.retrieve(c.text, 50))
                if d not in c.relevant and d not in c.cited
            ][: args.hard_negatives]
            for c in train_claims
        }
    pairs = build_pairs(
        train_claims,
        rationales,
        {d.id: d for d in documents},
        sentences,
        negative_docs=hard_negatives,
        seed=args.seed,
    )
    print(
        f"{len(train_claims)} train claims, {len(held)} hold-out; pairs {label_counts(pairs)}",
        file=sys.stderr,
    )
    held_gold = [c.label for c in held]

    held_candidates = (
        [scifact.unique_docs(retriever.retrieve(c.text, 50))[:3] for c in held] if retriever else []
    )

    def holdout_score(checkpoint: Path) -> float:
        verifier = NLIVerifier(str(checkpoint))
        if retriever is not None:  # open setting: abstract-level F1 over the top-3 retrieved
            preds = scifact.abstract_predictions(
                held, held_candidates, documents, verifier, threshold=args.threshold
            )
            return scifact.abstract_label_report(preds, held)["f1"]
        probs = scifact.oracle_probs(held, documents, verifier)
        report = scifact.sweep_threshold(held_gold, probs, [args.threshold])[args.threshold]
        return report["macro_f1"]

    out = Path(args.out)
    train_result = finetune(
        args.base_model,
        pairs,
        out,
        holdout_score,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        log=lambda m: print(m, file=sys.stderr),
    )
    # Threshold chosen on the hold-out only, for the base model and the fine-tuned one alike.
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    chosen = {}
    for name, path in (("base", args.base_model), ("fine_tuned", str(out / "best"))):
        verifier = NLIVerifier(path)
        if retriever is not None:
            cache: dict = {}
            sweep = {
                t: scifact.abstract_label_report(
                    scifact.abstract_predictions(
                        held, held_candidates, documents, verifier, threshold=t, cache=cache
                    ),
                    held,
                )["f1"]
                for t in grid
            }
            metric = "holdout_open_top3_f1"
        else:
            probs = scifact.oracle_probs(held, documents, verifier)
            sweep = {
                t: r["macro_f1"] for t, r in scifact.sweep_threshold(held_gold, probs, grid).items()
            }
            metric = "holdout_macro_f1"
        best_t = max(grid, key=lambda t: (sweep[t], -abs(t - 0.6)))
        chosen[name] = {
            "threshold": best_t,
            metric: sweep[best_t],
            f"{metric}_at_0.6": sweep[0.6],
        }
    result = {
        "base_model": args.base_model,
        "train_claims": len(train_claims),
        "holdout_claims": len(held),
        "pairs": label_counts(pairs),
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_len": 192,
        },
        **train_result,
        "thresholds_from_holdout": chosen,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_e2e(args: argparse.Namespace, settings: Settings) -> int:
    import tempfile
    from pathlib import Path

    from rag_fact_checking.evaluation import e2e

    with tempfile.TemporaryDirectory() as tmp:
        pipeline, embedder, setup, items = e2e.prepare_golden_pipeline(
            Path(tmp), verifier=args.verifier, base=settings
        )
        result = e2e.run_e2e(pipeline, items, embedder, setup)
    print(e2e.format_summary(result.summary))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(e2e.nan_free(result.to_dict()), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.update_baseline:
        if not args.baseline:
            raise ValueError("--update-baseline needs --baseline FILE")
        path = Path(args.baseline)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(e2e.baseline_from(result), indent=2) + "\n", encoding="utf-8")
        print(f"Baseline written to {path}")
        return 0
    if args.baseline:
        problems = e2e.compare_to_baseline(
            result, json.loads(Path(args.baseline).read_text(encoding="utf-8")), args.tolerance
        )
        if problems:
            print("REGRESSION:", *problems, sep="\n  - ", file=sys.stderr)
            return 1
        print(f"No regression against {args.baseline} (tolerance {args.tolerance}).")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Run: pip install 'rag-fact-checking[api]'", file=sys.stderr
        )
        return 1
    uvicorn.run("rag_fact_checking.api:app", host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragfc", description="Lightweight RAG fact-checking")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample", help="write the bundled demo corpus (no network)")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("ingest", help="fetch papers from arXiv into the corpus")
    s.add_argument("--topic", default="machine learning")
    s.add_argument("--max-results", type=int, default=10)
    s.add_argument("--full-text", action="store_true", help="download and parse PDFs (needs [pdf])")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("index", help="chunk + embed the corpus and build the index")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("ask", help="ask a question")
    s.add_argument("question")
    s.add_argument("--top-k", type=int, default=None)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("check", help="verify a claim against the corpus (needs RAGFC_VERIFIER=nli)")
    s.add_argument("claim")
    s.add_argument("--top-k", type=int, default=None)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("eval", help="compute retrieval metrics on a test set")
    s.add_argument("--testset", default=None, help="JSONL file (default: bundled sample test set)")
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser(
        "eval-scifact", help="benchmark retrieval and fact-checking on SciFact (needs [bench])"
    )
    s.add_argument("--split", choices=["train", "validation"], default="validation")
    s.add_argument("--limit", type=int, default=0, help="only the first N claims (smoke test)")
    s.add_argument("--threshold", type=float, default=0.6, help="NLI decision threshold")
    s.add_argument("--no-verify", action="store_true", help="retrieval experiment only")
    s.add_argument(
        "--extra-corpus",
        nargs="*",
        default=[],
        choices=["nfcorpus", "scidocs"],
        help="add unrelated abstracts (BEIR) to the corpus to test robustness to its size",
    )
    s.add_argument("--verifier-model", default=None, help="NLI model name or local checkpoint")
    s.add_argument("--data-dir", default=None, help="cache for the dataset and the indexes")
    s.add_argument("--output", default=None, help="also write the JSON result to this file")
    s.set_defaults(func=cmd_eval_scifact)

    s = sub.add_parser(
        "finetune-nli", help="fine-tune the NLI verifier on SciFact train (needs [ml,bench])"
    )
    s.add_argument("--base-model", default="cross-encoder/nli-MiniLM2-L6-H768")
    s.add_argument("--out", default="models/nli-scifact", help="checkpoint directory")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--lr", type=float, default=2e-5)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--holdout", type=float, default=0.15, help="share of train claims held out")
    s.add_argument(
        "--hard-negatives",
        type=int,
        default=0,
        help="retrieved non-evidence abstracts per claim used as neutral examples (0 = off)",
    )
    s.add_argument("--threshold", type=float, default=0.6, help="threshold for epoch selection")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--data-dir", default=None)
    s.add_argument("--output", default=None, help="also write the JSON report to this file")
    s.set_defaults(func=cmd_finetune_nli)

    s = sub.add_parser(
        "e2e", help="end-to-end evaluation on the bundled golden question set (no LLM judge)"
    )
    s.add_argument(
        "--verifier",
        choices=["overlap", "env", "none"],
        default="overlap",
        help="overlap: deterministic test double (CI); env: RAGFC_VERIFIER (e.g. the NLI model)",
    )
    s.add_argument("--baseline", default=None, help="JSON file with the reference scores")
    s.add_argument("--update-baseline", action="store_true", help="write the baseline and stop")
    s.add_argument("--tolerance", type=float, default=0.05, help="allowed drop per gated metric")
    s.add_argument("--output", default=None, help="write the full JSON result here")
    s.set_defaults(func=cmd_e2e)

    s = sub.add_parser("serve", help="start the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
