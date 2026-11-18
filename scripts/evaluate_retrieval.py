"""Measure compact retrieval on a labeled dev dataset without model calls."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_agent import RAGAgent
from rag_agent.retrieval import CANDIDATE_DEPTH, RERANK_DEPTH, ROUTINE_SEARCH_DEPTH, merge_rankings, search_rankings
from rag_preprocessing import build_index, pack_context


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def measure(dataset, reference, index):
    agent = RAGAgent(index, dataset["experts"], None, history=dataset["chat_history"])
    expected = {row["record_id"]: set(row["allowed_article_ids"]) for row in reference["requests"]}
    groups = defaultdict(list)
    for request in dataset["requests"]:
        group_id = request.get("group_id")
        key = ("group", group_id) if group_id is not None else ("single", request["record_id"])
        groups[key].append(request)

    rows_by_id = {}
    for group in groups.values():
        rankings = search_rankings(index, agent.group_query(group))
        candidates = {hit["article_id"] for ranking in rankings for hit in ranking}
        hits = merge_rankings([ranking[:agent.top_k] for ranking in rankings], agent.top_k)
        evidence = pack_context(index, hits, max_chars=agent.context_chars)
        selected = evidence[0] if evidence else None  # Same shared anchor as /run.
        chosen = f"{selected['article_id']} / {selected['section_id']}" if selected else None
        for request in group:
            record_id = request["record_id"]
            allowed = expected[record_id]
            rows_by_id[record_id] = {
                "record_id": record_id,
                "group_id": request.get("group_id"),
                "expert_id": request["expert_id"],
                "allowed_articles": sorted(allowed),
                "found_in_candidates": sorted(allowed & candidates),
                "missing_from_candidates": sorted(allowed - candidates),
                "candidate_articles": sorted(candidates),
                "selected_section": chosen,
                "selected_article_allowed": selected is not None and selected["article_id"] in allowed,
            }
    return [rows_by_id[r["record_id"]] for r in dataset["requests"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset.json"))
    parser.add_argument("--reference", type=Path, default=Path("data/reference.json"))
    parser.add_argument("--output", type=Path, default=Path("tmp/retrieval_evaluation"))
    args = parser.parse_args()
    dataset = read_json(args.dataset)
    dataset = dataset.get("dataset", dataset)
    started = time.monotonic()
    index = args.output / "index.sqlite3"
    build_index(dataset, index)
    rows = measure(dataset, read_json(args.reference), index)
    summary = {
        "requests": len(rows),
        "at_least_one_allowed_candidate": sum(bool(row["found_in_candidates"]) for row in rows),
        "all_allowed_candidates": sum(not row["missing_from_candidates"] for row in rows),
        "allowed_article_section_selected": sum(row["selected_article_allowed"] for row in rows),
        "allowed_candidates_found": sum(len(row["found_in_candidates"]) for row in rows),
        "allowed_candidates_total": sum(len(row["allowed_articles"]) for row in rows),
        "mean_candidate_articles": round(mean(len(row["candidate_articles"]) for row in rows), 2) if rows else 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (args.output / "results.json").write_text(
        json.dumps({"summary": summary, "requests": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Локальная проверка поиска compact",
        "",
        "| Метрика | Результат |",
        "|---|---:|",
        f"| Хотя бы одна допустимая статья среди кандидатов | {summary['at_least_one_allowed_candidate']}/{summary['requests']} |",
        f"| Все допустимые статьи среди кандидатов | {summary['all_allowed_candidates']}/{summary['requests']} |",
        f"| Допустимая статья выбрана для промпта | {summary['allowed_article_section_selected']}/{summary['requests']} |",
        f"| Найдено допустимых пар «запрос — статья» | {summary['allowed_candidates_found']}/{summary['allowed_candidates_total']} |",
        f"| Среднее число статей среди кандидатов | {summary['mean_candidate_articles']} |",
        "",
        f"Пул кандидатов собирается из выдач глубиной до {CANDIDATE_DEPTH} фрагментов на запрос. "
        f"Для дополнительного поиска по гигиене сна лимит {ROUTINE_SEARCH_DEPTH}, окно пересортировки {RERANK_DEPTH}. "
        "Эталон используется только после поиска.",
        "Выбранный раздел — общий источник, который /run передал бы экспертам группы. Генерация не запускалась.",
        "",
        "| № | ID запроса | Эксперт | Допустимые статьи от организаторов | Найдены среди кандидатов | Пропущены | Покрытие | Раздел для промпта | Выбор допустим? |",
        "|---:|---|---|---|---|---|---:|---|---|",
    ]
    for number, row in enumerate(rows, 1):
        allowed = ", ".join(row["allowed_articles"])
        found = ", ".join(row["found_in_candidates"]) or "—"
        missing = ", ".join(row["missing_from_candidates"]) or "—"
        coverage = f"{len(row['found_in_candidates'])}/{len(row['allowed_articles'])}"
        selected = row["selected_section"] or "—"
        status = "Да" if row["selected_article_allowed"] else "Нет"
        lines.append(
            f"| {number} | {row['record_id']} | {row['expert_id']} | {allowed} | {found} | {missing} | "
            f"{coverage} | {selected} | {status} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
