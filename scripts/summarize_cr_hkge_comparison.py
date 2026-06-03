"""Summarize KGAT vs CR-HKGE subset evaluation logs as a compact table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List


SCOPE_RE = re.compile(r"^(overall|enriched|standard):")
METRIC_RE = re.compile(r"(recall|precision|hit|ndcg)=\[([^\]]+)\]")


def parse_values(raw: str) -> List[float]:
    return [float(value) for value in raw.replace("\t", " ").split()]


def parse_log(path: Path) -> Dict[str, Dict[str, List[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    result: Dict[str, Dict[str, List[float]]] = {}

    for idx, line in enumerate(lines):
        scope_match = SCOPE_RE.match(line.strip())
        if not scope_match:
            continue

        scope = scope_match.group(1)
        if idx + 1 >= len(lines):
            raise RuntimeError("metrics line missing after %s in %s" % (scope, path))

        metrics = {
            name: parse_values(values)
            for name, values in METRIC_RE.findall(lines[idx + 1])
        }
        missing = {"recall", "precision", "hit", "ndcg"} - set(metrics)
        if missing:
            raise RuntimeError(
                "missing metrics %s for %s in %s" % (sorted(missing), scope, path)
            )
        result[scope] = metrics

    missing_scopes = {"overall", "enriched", "standard"} - set(result)
    if missing_scopes:
        raise RuntimeError("missing scopes %s in %s" % (sorted(missing_scopes), path))

    return result


def fmt(value: float) -> str:
    return "%.5f" % value


def fmt_delta(value: float) -> str:
    if value > 0:
        return "+%.5f" % value
    return "%.5f" % value


def row(scope: str, model_name: str, metrics: Dict[str, List[float]]) -> str:
    return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        scope,
        model_name,
        fmt(metrics["recall"][0]),
        fmt(metrics["ndcg"][0]),
        fmt(metrics["recall"][1]),
        fmt(metrics["ndcg"][1]),
        fmt(metrics["recall"][2]),
        fmt(metrics["ndcg"][2]),
    )


def delta_row(scope: str, kgat_metrics, cr_metrics) -> str:
    return "| %s | CR-HKGE - KGAT | %s | %s | %s | %s | %s | %s |" % (
        scope,
        fmt_delta(cr_metrics["recall"][0] - kgat_metrics["recall"][0]),
        fmt_delta(cr_metrics["ndcg"][0] - kgat_metrics["ndcg"][0]),
        fmt_delta(cr_metrics["recall"][1] - kgat_metrics["recall"][1]),
        fmt_delta(cr_metrics["ndcg"][1] - kgat_metrics["ndcg"][1]),
        fmt_delta(cr_metrics["recall"][2] - kgat_metrics["recall"][2]),
        fmt_delta(cr_metrics["ndcg"][2] - kgat_metrics["ndcg"][2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kgat-log", required=True)
    parser.add_argument("--cr-log", required=True)
    parser.add_argument("--kgat-name", default="KGAT")
    parser.add_argument("--cr-name", default="CR-HKGE")
    args = parser.parse_args()

    kgat = parse_log(Path(args.kgat_log))
    cr_hkge = parse_log(Path(args.cr_log))

    print("| Scope | Model | Recall@3 | NDCG@3 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for scope in ["overall", "enriched", "standard"]:
        print(row(scope, args.kgat_name, kgat[scope]))
        print(row(scope, args.cr_name, cr_hkge[scope]))
        print(delta_row(scope, kgat[scope], cr_hkge[scope]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
