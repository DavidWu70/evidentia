from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ENTRY_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+stage=(?P<stage>\S+)\s+meta=(?P<meta>\{.*\})$")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    w = rank - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _format_latency(values: list[float]) -> str:
    if not values:
        return "n/a"
    avg = sum(values) / len(values)
    p50 = _percentile(values, 50)
    p95 = _percentile(values, 95)
    return f"avg={avg:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms n={len(values)}"


def _format_rate(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{(100.0 * n / d):.1f}% ({n}/{d})"


def parse_log(path: Path) -> dict[str, Any]:
    filter_latencies: list[float] = []
    filter_yes_latencies: list[float] = []
    filter_no_latencies: list[float] = []
    filter_unknown_latencies: list[float] = []
    extract_latencies: list[float] = []
    parse_error_invalid_json_count = 0
    pending_filter_ms: float | None = None

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ENTRY_RE.match(line.strip())
        if not match:
            continue
        stage = match.group("stage")
        try:
            meta = json.loads(match.group("meta"))
        except Exception:
            meta = {}

        if stage == "filter_inference_end":
            elapsed = float(meta.get("elapsed_ms", 0.0) or 0.0)
            if elapsed > 0:
                filter_latencies.append(elapsed)
                pending_filter_ms = elapsed
            continue

        if stage == "filter_raw_output":
            decision = str(meta.get("decision", "")).strip().lower()
            if pending_filter_ms is not None:
                if decision == "yes":
                    filter_yes_latencies.append(pending_filter_ms)
                elif decision == "no":
                    filter_no_latencies.append(pending_filter_ms)
                else:
                    filter_unknown_latencies.append(pending_filter_ms)
            pending_filter_ms = None
            continue

        if stage == "extract_inference_end":
            elapsed = float(meta.get("elapsed_ms", 0.0) or 0.0)
            if elapsed > 0:
                extract_latencies.append(elapsed)
            continue

        if stage == "parse_error_invalid_json":
            parse_error_invalid_json_count += 1

    extract_calls = len(extract_latencies)
    json_valid_calls = max(0, extract_calls - parse_error_invalid_json_count)

    return {
        "filter_latencies": filter_latencies,
        "filter_yes_latencies": filter_yes_latencies,
        "filter_no_latencies": filter_no_latencies,
        "filter_unknown_latencies": filter_unknown_latencies,
        "extract_latencies": extract_latencies,
        "parse_error_invalid_json_count": parse_error_invalid_json_count,
        "extract_calls": extract_calls,
        "json_valid_calls": json_valid_calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize MedGemma debug timings from evidentia_medgemma_raw.log"
    )
    parser.add_argument(
        "--log-path",
        default="/tmp/evidentia_medgemma_raw.log",
        help="Path to MedGemma raw debug log (default: /tmp/evidentia_medgemma_raw.log)",
    )
    parser.add_argument(
        "--list-latencies",
        action="store_true",
        help="Print each filter/extract latency entry in addition to summary.",
    )
    args = parser.parse_args()

    path = Path(args.log_path).expanduser()
    if not path.exists():
        raise SystemExit(f"log file not found: {path}")

    stats = parse_log(path)

    print(f"log_path: {path}")
    print(f"filter_latency: {_format_latency(stats['filter_latencies'])}")
    print(f"filter_yes_latency: {_format_latency(stats['filter_yes_latencies'])}")
    print(f"filter_no_latency: {_format_latency(stats['filter_no_latencies'])}")
    print(f"extract_latency: {_format_latency(stats['extract_latencies'])}")
    print(
        "extract_json_valid_rate: "
        + _format_rate(stats["json_valid_calls"], stats["extract_calls"])
    )
    print(
        "extract_invalid_json_count: "
        + str(stats["parse_error_invalid_json_count"])
    )

    if args.list_latencies:
        print("filter_latency_values_ms:", ", ".join(f"{v:.2f}" for v in stats["filter_latencies"]) or "n/a")
        print("extract_latency_values_ms:", ", ".join(f"{v:.2f}" for v in stats["extract_latencies"]) or "n/a")


if __name__ == "__main__":
    main()
