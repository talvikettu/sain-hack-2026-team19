"""Explicit, checkpointed local demo using the first N organizer examples."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from rag_preprocessing import build_index
from .agent import RAGAgent
from .client import ChatClient


def write_reports(directory, run):
    directory.mkdir(parents=True, exist_ok=True)
    def save(name, content):
        temp = directory / (name + ".tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(directory / name)
    save("run.json", json.dumps(run, ensure_ascii=False, indent=2))
    answers = [r["answer"] for r in run["results"] if "answer" in r]
    save("answers.json", json.dumps({"requests": answers}, ensure_ascii=False, indent=2))
    lines = ["# Пробный прогон RAG: первые запросы dev", "", "Локальный отчёт. Не публиковать данные и ответы в Git.",
             "", f"LLM calls: {run['calls']}; total tokens: {run['total_tokens']}",
             "", "Структура и источники проверены кодом. Медицинская проверка и оценка организаторов не выполнялись."]
    for i, result in enumerate(run["results"], 1):
        lines += ["", f"## {i}. {result['expert_id']}", "", "**Вопрос:** " + result["question"], ""]
        if "answer" not in result:
            lines += ["Ошибка: " + result["error_type"]]
            continue
        answer = result["answer"]
        tokens = sum(x["usage"]["total_tokens"] for x in result["attempts"])
        lines += [answer["text"], "", f"Отказ: {answer['refused']}; токены: {tokens}; время: {result['seconds']} с.",
                  "", "Цитаты: " + ", ".join(c["article_id"] + "/" + c["section_id"] for c in answer["citations"]),
                  "", "Флаги ручной проверки: " + str(result["checks"]["advisory_flags"])]
    save("report.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Real LLM demo; consumes organizer quota")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", default="tmp/rag_demo")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--dry-run", action="store_true", help="Prepare inspectable payloads; no API calls or credentials")
    parser.add_argument("--no-history", action="store_true", help="Do not send any conversation history")
    parser.add_argument("--profile", choices=("compact", "full"), default="compact")
    parser.add_argument("--token-budget", type=int, default=120000)
    args = parser.parse_args()
    if not 1 <= args.limit <= 60 or args.timeout <= 0 or args.token_budget < 1:
        parser.error("Demo supports 1–60 requests, a positive timeout and token budget")
    output = Path(args.output)
    if not args.dry_run and (output / "run.json").exists():
        parser.error("Output already contains a run; select a new directory to avoid accidental repeat charges")
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8-sig"))
    dataset = data.get("dataset", data)
    deadline = time.monotonic() + args.timeout
    client = None if args.dry_run else ChatClient.from_env(max_calls=args.limit + 2, token_budget=args.token_budget)
    index = output / "index.sqlite3"
    index_report = build_index(dataset, index, deadline=deadline)
    agent = RAGAgent(index, dataset["experts"], client,
                     history=[] if args.no_history else dataset["chat_history"], prompt_profile=args.profile)
    if args.dry_run:
        requests = []
        for record in dataset["requests"][:args.limit]:
            prepared = agent.prepare(record["utterance"]["text"], record["expert_id"],
                                     user_id=record["user_id"], record_id=record["record_id"],
                                     at=record["utterance"]["at"], deadline=deadline)
            requests.append({"record_id": record["record_id"], "expert_id": record["expert_id"],
                             "history_messages": prepared["history_messages"],
                             "evidence_sections": len(prepared["allowed"]), "messages": prepared["messages"]})
        output.mkdir(parents=True, exist_ok=True)
        (output / "preview.json").write_text(json.dumps({"dry_run": True, "calls": 0,
                 "history_enabled": not args.no_history, "profile": args.profile,
                 "requests": requests}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared {len(requests)} requests; zero API calls. Inspect preview.json locally.")
        return 0
    run = {"started_at": datetime.now(timezone.utc).isoformat(), "model": client.model,
           "profile": args.profile,
           "index_build_seconds": index_report["build_seconds"],
           "limit": args.limit, "calls": 0, "total_tokens": 0, "results": []}
    write_reports(output, run)
    for number, record in enumerate(dataset["requests"][:args.limit], 1):
        try:
            result = agent.answer(record["utterance"]["text"], record["expert_id"],
                                  user_id=record["user_id"], record_id=record["record_id"],
                                  at=record["utterance"]["at"], deadline=deadline)
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            result = {"record_id": record["record_id"], "expert_id": record["expert_id"],
                      "question": record["utterance"]["text"], "error_type": type(error).__name__}
        run["results"].append(result)
        run["calls"], run["total_tokens"] = client.calls, client.total_tokens
        write_reports(output, run)
        status = "ok" if "answer" in result else result["error_type"]
        print(f"{number}/{args.limit}: {status}; calls={client.calls}; tokens={client.total_tokens}", flush=True)
        if "error_type" in result:
            # Stop on quota/network/schema failures rather than repeatedly spending.
            break
    successful = sum("answer" in r for r in run["results"])
    print(f"Finished: {successful}/{args.limit} validated answers. Local report saved.", flush=True)
    return 0 if successful == args.limit else 1


if __name__ == "__main__":
    raise SystemExit(main())
