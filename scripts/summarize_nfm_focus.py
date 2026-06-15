"""Summarize CR-HKGE tuning candidates against NFM.

Inputnya adalah file `*_subset_eval.log` yang dihasilkan oleh
`run_cr_hkge_final_study.sh`. Output dibuat ringkas agar mudah melihat apakah
kandidat CR-HKGE mendekati atau mengalahkan NFM pada metrik utama.
"""

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
        if not match or idx + 1 >= len(lines):
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


def model_name(path: Path) -> str:
    return path.name.replace("_subset_eval.log", "")


def print_scope(scope: str, rows: list[tuple[str, dict[str, dict[str, list[float]]]]], reference: str) -> None:
    parsed_by_name = {name: parsed for name, parsed in rows}
    ref = parsed_by_name.get(reference, {}).get(scope)

    print("\n## %s vs %s" % (scope.capitalize(), reference))
    print("| Model | Recall@3 | dR@3 | Hit@3 | dHit@3 | NDCG@3 | dN@3 | Recall@10 | dR@10 | NDCG@10 | dN@10 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def sort_key(row: tuple[str, dict[str, dict[str, list[float]]]]) -> tuple[float, float]:
        metrics = row[1].get(scope)
        if not metrics:
            return (-1.0, -1.0)
        return (metrics["ndcg"][0], metrics["recall"][2])

    for name, parsed in sorted(rows, key=sort_key, reverse=True):
        metrics = parsed.get(scope)
        if not metrics:
            continue
        if ref:
            deltas = {
                "r3": metrics["recall"][0] - ref["recall"][0],
                "hit3": metrics["hit"][0] - ref["hit"][0],
                "n3": metrics["ndcg"][0] - ref["ndcg"][0],
                "r10": metrics["recall"][2] - ref["recall"][2],
                "n10": metrics["ndcg"][2] - ref["ndcg"][2],
            }
        else:
            deltas = {key: 0.0 for key in ["r3", "hit3", "n3", "r10", "n10"]}

        print("| %s | %s | %+0.5f | %s | %+0.5f | %s | %+0.5f | %s | %+0.5f | %s | %+0.5f |" % (
            name,
            fmt(metrics["recall"][0]),
            deltas["r3"],
            fmt(metrics["hit"][0]),
            deltas["hit3"],
            fmt(metrics["ndcg"][0]),
            deltas["n3"],
            fmt(metrics["recall"][2]),
            deltas["r10"],
            fmt(metrics["ndcg"][2]),
            deltas["n10"],
        ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--reference", default="nfm")
    args = parser.parse_args()

    rows = [(model_name(Path(raw)), parse_log(Path(raw))) for raw in args.logs]
    for scope in ["overall", "enriched", "standard"]:
        print_scope(scope, rows, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
