"""Detectability vs text length and temperature: SynthID-Text (Tournament)
vs. Gumbel sampling -- reproduces the shape of the paper's Fig. 3a at
smaller scale.

For each temperature, generates ONE long response per prompt per method
(tournament-watermarked, gumbel-watermarked, unwatermarked), then
truncates each to every requested length and scores at that length --
this avoids regenerating from scratch for every length bucket.

Usage:
    uv run python experiments/run_length_temperature_sweep.py \
        --model gpt2 --n_prompts 15 \
        --lengths 50,100,200 --temperatures 0.7,1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textsus.evaluation.metrics import tpr_at_fpr
from textsus.generation.watermarked_generator import WatermarkedGenerator
from textsus.sampling.gumbel import gumbel_mean_score
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
    parser.add_argument("--lengths", type=str, default="50,100,200")
    parser.add_argument("--temperatures", type=str, default="0.7")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="results/tables/length_temperature_sweep.json")
    parser.add_argument("--fig_dir", type=str, default="results/figures")
    args = parser.parse_args()

    lengths = sorted(int(x) for x in args.lengths.split(","))
    temperatures = [float(x) for x in args.temperatures.split(",")]
    max_length = max(lengths)

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

    all_results = {}

    for temp in temperatures:
        print(f"\n=== temperature={temp} ===")
        generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=args.m, temperature=temp)

        # Generate once at max_length per prompt per method; truncate for shorter lengths.
        tournament_runs, gumbel_runs, uwm_runs = [], [], []
        for i, prompt in enumerate(prompts):
            t_wm = generator.generate(prompt, max_new_tokens=max_length, watermark=True,
                                       method="tournament", ignore_eos=True)
            g_wm = generator.generate(prompt, max_new_tokens=max_length, watermark=True,
                                       method="gumbel", ignore_eos=True)
            uwm = generator.generate(prompt, max_new_tokens=max_length, watermark=False,
                                      ignore_eos=True)
            tournament_runs.append(t_wm.token_ids)
            gumbel_runs.append(g_wm.token_ids)
            uwm_runs.append(uwm.token_ids)
            print(f"  [{i+1}/{len(prompts)}] generated")

        length_results = {}
        for length in lengths:
            t_scores = [mean_score(toks[:length], args.key, num_layers=args.m)
                        for toks in tournament_runs if len(toks) >= length]
            g_scores = [gumbel_mean_score(toks[:length], args.key)
                        for toks in gumbel_runs if len(toks) >= length]
            u_scores_for_t = [mean_score(toks[:length], args.key, num_layers=args.m)
                               for toks in uwm_runs if len(toks) >= length]
            u_scores_for_g = [gumbel_mean_score(toks[:length], args.key)
                               for toks in uwm_runs if len(toks) >= length]

            t_tpr, t_thresh = tpr_at_fpr(t_scores, u_scores_for_t, fpr=args.fpr)
            g_tpr, g_thresh = tpr_at_fpr(g_scores, u_scores_for_g, fpr=args.fpr)

            length_results[length] = {
                "tournament_tpr": t_tpr,
                "tournament_threshold": t_thresh,
                "gumbel_tpr": g_tpr,
                "gumbel_threshold": g_thresh,
                "n_samples": len(t_scores),
            }
            print(f"  length={length:>4}  tournament_tpr={t_tpr:.3f}  gumbel_tpr={g_tpr:.3f}")

        all_results[temp] = length_results

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {out_path}")

    # Plot: one figure per temperature, TPR@FPR vs length, tournament vs gumbel.
    try:
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        for temp, length_results in all_results.items():
            xs = sorted(length_results.keys())
            t_ys = [length_results[x]["tournament_tpr"] for x in xs]
            g_ys = [length_results[x]["gumbel_tpr"] for x in xs]

            plt.figure(figsize=(5, 4))
            plt.plot(xs, t_ys, marker="o", label="SynthID-Text (Tournament)")
            plt.plot(xs, g_ys, marker="o", label="Gumbel sampling")
            plt.xlabel("Number of tokens")
            plt.ylabel(f"TPR @ FPR={args.fpr}")
            plt.title(f"Detectability vs length (temperature={temp})")
            plt.ylim(0, 1.05)
            plt.legend()
            plt.tight_layout()
            fig_path = fig_dir / f"detectability_temp_{temp}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Figure saved to {fig_path}")
    except ImportError:
        print("matplotlib not installed -- skipping figure generation (JSON results still saved).")


if __name__ == "__main__":
    main()