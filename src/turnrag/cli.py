from __future__ import annotations

import argparse
import sys
import time

from turnrag.config import COLLECTIONS, load_config


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _dev_sample(cfg, size: int) -> list[str]:
    from turnrag import data

    tasks = data.load_tasks(cfg, "dev")
    answerable = [t for t in tasks if t.answerability in ("ANSWERABLE", "PARTIAL")]
    step = max(1, len(answerable) // size)
    picked = [task.task_id for task in answerable[::step][:size]]
    return picked + [t.task_id for t in tasks if t.answerability not in ("ANSWERABLE", "PARTIAL")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="turnrag")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download the MTRAG files and unpack the corpora")
    models = commands.add_parser("models", help="download the pinned model files")
    models.add_argument("--export", action="store_true", help="also the weights needed by export")
    models.add_argument("--eval", action="store_true", help="also the BERTScore model for scoring")
    commands.add_parser("export", help="export the sparse document encoder to ONNX int8")
    index = commands.add_parser("index", help="build the BM25 and learned sparse indexes")
    index.add_argument("--collections", nargs="+", choices=COLLECTIONS)
    index.add_argument("--legs", nargs="+", default=["bm25", "sparse"], choices=["bm25", "sparse"])
    for name in ("rewrites", "retrieval", "answers", "judge", "score"):
        sub = commands.add_parser(name)
        sub.add_argument("--split", choices=["dev", "test"], required=True)
        if name == "rewrites":
            sub.add_argument("--workers", type=int, default=2)
        if name == "answers":
            sub.add_argument("--sample", type=int, default=0)
        if name == "score":
            sub.add_argument("--bf16", action="store_true")
    commands.add_parser("gate", help="fit the answer / I don't know gate on dev")
    check = commands.add_parser("check-metrics", help="compare RB_alg parts with IBM's values")
    check.add_argument("--sample", type=int, default=200)
    check.add_argument("--bf16", action="store_true")
    judge = commands.add_parser("check-judge", help="compare the local IDK judge with IBM's")
    judge.add_argument("--sample", type=int, default=300)
    chat = commands.add_parser("chat", help="chat in the terminal")
    chat.add_argument("--collection", choices=COLLECTIONS, required=True)
    commands.add_parser("serve", help="run the HTTP API and the chat page")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "fetch":
        from turnrag import data

        data.download(cfg, log=_log)
    elif args.command == "models":
        from turnrag import data

        groups = ("run",) + (("export",) if args.export else ()) + (("eval",) if args.eval else ())
        data.download_models(cfg, groups, log=_log)
    elif args.command == "export":
        from turnrag.export import export_sparse

        export_sparse(cfg, log=_log)
    elif args.command == "index":
        from turnrag.build import build_all

        build_all(cfg, args.collections or cfg.data.collections, args.legs, log=_log)
    elif args.command == "rewrites":
        from turnrag import data
        from turnrag.evaluate import compute_variants

        compute_variants(cfg, data.load_tasks(cfg, args.split), args.split, args.workers, _log)
    elif args.command == "retrieval":
        from turnrag.evaluate import retrieval_report

        retrieval_report(cfg, args.split, log=_log)
    elif args.command == "gate":
        from turnrag.evaluate import fit_gate

        fit_gate(cfg, log=_log)
    elif args.command == "answers":
        from turnrag.evaluate import run_answers

        ids = _dev_sample(cfg, args.sample) if args.sample else None
        run_answers(cfg, args.split, ids, log=_log)
    elif args.command == "judge":
        from turnrag.evaluate import judge_answers

        judge_answers(cfg, args.split, log=_log)
    elif args.command == "score":
        from turnrag.evaluate import score_answers

        score_answers(cfg, args.split, args.bf16, log=_log)
    elif args.command == "check-metrics":
        from turnrag.evaluate import metric_check

        metric_check(cfg, args.sample, args.bf16, log=_log)
    elif args.command == "check-judge":
        from turnrag.evaluate import judge_check

        judge_check(cfg, args.sample, log=_log)
    elif args.command == "chat":
        _chat(cfg, args.collection)
    elif args.command == "serve":
        import uvicorn

        from turnrag.api import create_app

        uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port)
    return 0


def _chat(cfg, collection: str) -> None:
    from turnrag.data import Turn
    from turnrag.evaluate import pipeline

    engine = pipeline(cfg)
    turns: list[Turn] = []
    print(f"Chatting over the {collection} collection. Empty line to quit.")
    while True:
        try:
            question = input("you> ").strip()
        except EOFError:
            break
        if not question:
            break
        turns.append(Turn("user", question))
        pieces: list[str] = []
        passages: list[dict] = []
        for event in engine.stream(collection, turns):
            if event["type"] == "plan":
                passages = event["passages"]
                print(f"[{event['action']}] {event['rewrite'] or ''}")
                print("bot> ", end="", flush=True)
            elif event["type"] == "token":
                pieces.append(event["text"])
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                print()
                for number, passage in enumerate(passages, start=1):
                    print(f"  [{number}] {passage['title'] or passage['document_id']}")
        turns.append(Turn("agent", "".join(pieces).strip()))


if __name__ == "__main__":
    sys.exit(main())
