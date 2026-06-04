"""Summarize final CR-HKGE study logs into markdown tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


SCOPE_RE = re.compile(r"^(overall|enriched|standard):")
METRIC_RE = re.compile(r"(recall|precision|hit|ndcg)=\[([^\]]+)\]")


def parse_values(raw: str) -> list[float]:
    return [float(value) for value in raw.replace("\t", " ").split()]


def parse_log(path: Path) -> dict[str, dict[str, list[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result: dict[str, dict[str, list[float]]] = {}
    for idx, line in enumerate(lines):
        match = SCOPE_RE.match(line.strip())
        if not match:
            continue
        if idx + 1 >= len(lines):
            continue
        metrics = {
            name: parse_values(values)
            for name, values in METRIC_RE.findall(lines[idx + 1])
        }
        if metrics:
            result[match.group(1)] = metrics
    return result


def fmt(value: float) -> str:
    return "%.5f" % value


def print_table(scope: str, rows: list[tuple[str, dict[str, dict[str, list[float]]]]]) -> None:
    print("\n## %s" % scope.capitalize())
    print("| Model | Recall@3 | Precision@3 | Hit@3 | NDCG@3 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, parsed in rows:
        metrics = parsed.get(scope)
        if not metrics:
            continue
        print("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name,
            fmt(metrics["recall"][0]),
            fmt(metrics["precision"][0]),
            fmt(metrics["hit"][0]),
            fmt(metrics["ndcg"][0]),
            fmt(metrics["recall"][1]),
            fmt(metrics["ndcg"][1]),
            fmt(metrics["recall"][2]),
            fmt(metrics["ndcg"][2]),
        ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    rows = []
    for raw in args.logs:
        path = Path(raw)
        name = path.name.replace("_subset_eval.log", "")
        rows.append((name, parse_log(path)))

    for scope in ["overall", "enriched", "standard"]:
        print_table(scope, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
