"""Ablation: how does the number of Tournament sampling layers (m) affect
detectability? Reproduces the paper's claim that detectability improves
with more layers but with diminishing returns (each layer uses up some
of the available entropy, and the marginal benefit shrinks deeper into
the tournament).

Note: m changes the actual sampling behavior (N = 2**m candidates compete
each step), not just the scoring -- so each m value needs its own
generation run, unlike the length sweep which could truncate a single
long generation.

Usage:
    uv run python experiments/run_layer_ablation.py \
        --model gpt2 --n_prompts 15 --layers 1,2,4,6,8 --max_new_tokens 100
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from textsus.evaluation.metrics import tpr_at_fpr
from textsus.generation.watermarked_generator import WatermarkedGenerator
from textsus.scoring.mean_score import mean_score


def load_prompts(path: str, n: int) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompts_file", type=str, default="data/eli5_test.jsonl")
    parser.add_argument("--n_prompts", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--layers", type=str, default="1,2,4,6,8")
    parser.add_argument("--top_k", type=int, default=100,
                         help="LLM candidate pool size. Should comfortably exceed "
                              "2**max(layers) or duplicate candidates will saturate "
                              "the tournament and distort the layer-count trend.")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="results/tables/layer_ablation.json")
    parser.add_argument("--fig_dir", type=str, default="results/figures")
    args = parser.parse_args()

    layer_values = sorted(int(x) for x in args.layers.split(","))

    if Path(args.prompts_file).exists():
        prompts = load_prompts(args.prompts_file, args.n_prompts)
    else:
        print(f"{args.prompts_file} not found -- using a few built-in prompts instead.")
        prompts = [
            "Why is the sky blue?",
            "How do vaccines work?",
            "Why do we dream?",
            "How does the internet work?",
        ][: args.n_prompts]

    # Unwatermarked baseline: one generation per prompt, independent of m.
    # Uses m=1 generator purely to call .generate(); watermark=False ignores m.
    base_generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=1, top_k=args.top_k)
    uwm_token_runs = []
    for i, prompt in enumerate(prompts):
        uwm = base_generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=False)
        uwm_token_runs.append(uwm.token_ids)
        print(f"[baseline {i+1}/{len(prompts)}] generated, len={len(uwm.token_ids)}")

    results = {}

    for m in layer_values:
        print(f"\n=== m={m} (N={2**m} candidates/step, top_k={args.top_k}) ===")
        generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=m, top_k=args.top_k)

        wm_scores, uwm_scores = [], []
        for i, prompt in enumerate(prompts):
            wm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True, method="tournament")
            if not wm.token_ids or not uwm_token_runs[i]:
                continue
            wm_scores.append(mean_score(wm.token_ids, args.key, num_layers=m))
            uwm_scores.append(mean_score(uwm_token_runs[i], args.key, num_layers=m))
            print(f"  [{i+1}/{len(prompts)}] wm_score={wm_scores[-1]:.3f} uwm_score={uwm_scores[-1]:.3f}")

        tpr, threshold = tpr_at_fpr(wm_scores, uwm_scores, fpr=args.fpr)
        results[m] = {
            "mean_wm_score": statistics.mean(wm_scores) if wm_scores else float("nan"),
            "mean_uwm_score": statistics.mean(uwm_scores) if uwm_scores else float("nan"),
            "tpr_at_fpr": tpr,
            "threshold": threshold,
            "n_samples": len(wm_scores),
        }
        print(f"  m={m}: mean_wm_score={results[m]['mean_wm_score']:.3f}  TPR@FPR={args.fpr}={tpr:.3f}")

    output = {"top_k": args.top_k, "max_new_tokens": args.max_new_tokens, "results": results}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {out_path}")

    try:
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        xs = sorted(results.keys())
        mean_scores = [results[m]["mean_wm_score"] for m in xs]
        tprs = [results[m]["tpr_at_fpr"] for m in xs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(xs, mean_scores, marker="o", color="tab:blue")
        ax1.set_xlabel("Number of tournament layers (m)")
        ax1.set_ylabel("Mean watermark score")
        ax1.set_title("Score vs. layers (diminishing returns)")

        ax2.plot(xs, tprs, marker="o", color="tab:orange")
        ax2.set_xlabel("Number of tournament layers (m)")
        ax2.set_ylabel(f"TPR @ FPR={args.fpr}")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Detectability vs. layers")

        plt.tight_layout()
        fig_path = fig_dir / "layer_ablation.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Figure saved to {fig_path}")
    except ImportError:
        print("matplotlib not installed -- skipping figure generation (JSON results still saved).")


if __name__ == "__main__":
    main()