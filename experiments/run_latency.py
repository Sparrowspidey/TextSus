"""Latency benchmark: ms/token for each sampling method, and the % overhead
each watermarking scheme adds over plain (unwatermarked) generation --
mirrors the paper's latency comparison (their Table: SynthID-Text +0.57%,
Gumbel +0.26%, Soft Red List +0.28% over an unwatermarked baseline).

Usage:
    uv run python experiments/run_latency.py --model gpt2 --max_new_tokens 100 --n_runs 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from textsus.generation.watermarked_generator import WatermarkedGenerator

METHODS = [
    ("unwatermarked", False, "tournament"),
    ("tournament", True, "tournament"),
    ("gumbel", True, "gumbel"),
    ("soft_red_list", True, "soft_red_list"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompt", type=str, default="Why is the sky blue?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=3, help="timed repeats per method")
    parser.add_argument("--n_warmup", type=int, default=10, help="untimed warm-up tokens")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--out", type=str, default="results/tables/latency.json")
    args = parser.parse_args()

    generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=args.m)

    results = {}

    for label, watermark, method in METHODS:
        # Untimed warm-up: absorbs first-call overhead (lazy init, kv-cache
        # allocation, etc.) so it doesn't skew the first method measured.
        generator.generate(
            args.prompt, max_new_tokens=args.n_warmup, watermark=watermark,
            method=method, ignore_eos=True,
        )

        ms_per_token_runs = []
        for _ in range(args.n_runs):
            start = time.perf_counter()
            result = generator.generate(
                args.prompt, max_new_tokens=args.max_new_tokens, watermark=watermark,
                method=method, ignore_eos=True,
            )
            elapsed = time.perf_counter() - start
            ms_per_token_runs.append(1000 * elapsed / len(result.token_ids))

        mean_ms = statistics.mean(ms_per_token_runs)
        std_ms = statistics.pstdev(ms_per_token_runs) if len(ms_per_token_runs) > 1 else 0.0

        results[label] = {
            "mean_ms_per_token": mean_ms,
            "std_ms_per_token": std_ms,
            "n_runs": args.n_runs,
        }
        print(f"{label:>15}: {mean_ms:.3f} ms/token (+/- {std_ms:.3f}, n={args.n_runs})")

    baseline = results["unwatermarked"]["mean_ms_per_token"]
    for label, stats in results.items():
        stats["overhead_pct_vs_unwatermarked"] = 100 * (stats["mean_ms_per_token"] - baseline) / baseline

    print("\nOverhead vs. unwatermarked generation:")
    for label, stats in results.items():
        print(f"  {label:>15}: {stats['overhead_pct_vs_unwatermarked']:+.2f}%")

    out = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "n_runs": args.n_runs,
        "m": args.m,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()