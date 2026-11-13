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


def cmd_eval_scifact(args: argparse.Namespace, settings: Settings) -> int:
    import dataclasses
    import re
    from datetime import date
    from pathlib import Path

    from rag_fact_checking.corpus import build_index, save_documents
    from rag_fact_checking.embeddings import build_embedder
    from rag_fact_checking.evaluation import scifact
    from rag_fact_checking.index import KnowledgeBase, KnowledgeBaseNotFound
    from rag_fact_checking.retriever import HybridRetriever
    from rag_fact_checking.verifier import build_verifier

    root = Path(args.data_dir) if args.data_dir else settings.data_dir / "scifact"
    documents = scifact.build_documents(scifact.read_parquet(scifact.download(root, "corpus")))
    rows = scifact.read_parquet(scifact.download(root, args.split))
    claims = scifact.build_claims(rows)
    if args.limit:
        claims = claims[: args.limit]

    embedder = build_embedder(settings)
    # Abstracts have no bibliography, so the reference heuristics are switched off.
    run = dataclasses.replace(
        settings,
        data_dir=root / ("run-" + re.sub(r"[^A-Za-z0-9._-]+", "_", embedder.name)),
        drop_references=False,
    )
    try:
        KnowledgeBase.load(run.index_dir, expected_embedder=embedder.name)
    except (KnowledgeBaseNotFound, ValueError):
        print(f"Indexing {len(documents)} abstracts with {embedder.name} ...", file=sys.stderr)
        save_documents(documents, run.corpus_path)
        build_index(run, embedder)
    retriever = HybridRetriever.from_settings(run, embedder)

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
            "documents": len(documents),
        },
        **result,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
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
    s.add_argument("--data-dir", default=None, help="cache for the dataset and the indexes")
    s.add_argument("--output", default=None, help="also write the JSON result to this file")
    s.set_defaults(func=cmd_eval_scifact)

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
