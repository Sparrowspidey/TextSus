"""Distortionary SynthID-Text (Tournament, competitors_per_match > 2) vs.
Soft Red List: detectability-vs-quality trade-off, reproducing the shape
of the paper's Fig. 3c.

Both methods have a strength knob that trades text quality for stronger
detectability:
  - Tournament:     competitors_per_match (more competitors = stronger)
  - Soft Red List:  delta (bigger logit bias = stronger)

For each strength setting of each method, generates watermarked text,
computes TPR@FPR against a shared unwatermarked baseline, and computes
mean perplexity as the quality proxy (paper uses log-perplexity).

Usage:
    uv run python experiments/run_distortionary_comparison.py \
        --model gpt2 --n_prompts 10 --max_new_tokens 80 \
        --tournament_strengths 2,3,4 --soft_red_list_deltas 0,1,2,4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from textsus.evaluation.metrics import tpr_at_fpr
from textsus.evaluation.perplexity import compute_perplexity
from textsus.generation.watermarked_generator import WatermarkedGenerator
from textsus.sampling.soft_red_list import z_score
from textsus.scoring.mean_score import mean_score


def load_prompts(path: str, n: int) -> list[str]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def mean_log_perplexity(model, tokenizer, texts, device) -> float:
    vals = []
    for t in texts:
        ppl = compute_perplexity(model, tokenizer, t, device)
        if ppl == ppl and ppl > 0:  # not NaN
            vals.append(math.log(ppl))
    return statistics.mean(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompts_file", type=str, default="data/eli5_test.jsonl")
    parser.add_argument("--n_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--tournament_layers", type=int, default=3,
                         help="num_layers for distortionary Tournament (N = strength**layers candidates)")
    parser.add_argument("--tournament_strengths", type=str, default="2,3,4",
                         help="competitors_per_match values to test (2 = non-distortionary baseline)")
    parser.add_argument("--soft_red_list_deltas", type=str, default="0,1,2,4")
    parser.add_argument("--top_k", type=int, default=500,
                         help="should comfortably exceed max(tournament_strengths)**tournament_layers")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="results/tables/distortionary_comparison.json")
    parser.add_argument("--fig_dir", type=str, default="results/figures")
    args = parser.parse_args()

    tournament_strengths = [int(x) for x in args.tournament_strengths.split(",")]
    soft_red_list_deltas = [float(x) for x in args.soft_red_list_deltas.split(",")]

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

    # Shared unwatermarked baseline (independent of both methods' strength settings).
    base_generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=1, top_k=args.top_k)
    uwm_runs = []
    for i, prompt in enumerate(prompts):
        uwm = base_generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=False)
        uwm_runs.append(uwm)
        print(f"[baseline {i+1}/{len(prompts)}] generated")
    uwm_texts = [r.text for r in uwm_runs]
    uwm_tournament_scores = [mean_score(r.token_ids, args.key, num_layers=args.tournament_layers)
                              for r in uwm_runs if r.token_ids]
    uwm_green_fractions = [z_score(r.token_ids, args.key) for r in uwm_runs if r.token_ids]
    uwm_mean_log_ppl = mean_log_perplexity(base_generator.model, base_generator.tokenizer,
                                            uwm_texts, base_generator.device)

    results = {"tournament": {}, "soft_red_list": {}, "unwatermarked_log_perplexity": uwm_mean_log_ppl}

    # --- Tournament (distortionary) sweep ---
    print("\n--- Tournament (distortionary) ---")
    for strength in tournament_strengths:
        n_candidates = strength**args.tournament_layers
        print(f"\ncompetitors_per_match={strength} (N={n_candidates} candidates/step)")
        generator = WatermarkedGenerator(
            model_name=args.model, key=args.key, m=args.tournament_layers,
            top_k=args.top_k, competitors_per_match=strength,
        )
        wm_scores, wm_texts = [], []
        for i, prompt in enumerate(prompts):
            wm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True, method="tournament")
            if not wm.token_ids:
                continue
            wm_scores.append(mean_score(wm.token_ids, args.key, num_layers=args.tournament_layers))
            wm_texts.append(wm.text)
            print(f"  [{i+1}/{len(prompts)}] score={wm_scores[-1]:.3f}")

        tpr, threshold = tpr_at_fpr(wm_scores, uwm_tournament_scores, fpr=args.fpr)
        log_ppl = mean_log_perplexity(generator.model, generator.tokenizer, wm_texts, generator.device)

        results["tournament"][strength] = {
            "mean_score": statistics.mean(wm_scores) if wm_scores else float("nan"),
            "tpr_at_fpr": tpr,
            "mean_log_perplexity": log_ppl,
            "n_samples": len(wm_scores),
        }
        print(f"  strength={strength}: TPR@FPR={args.fpr}={tpr:.3f}  mean_log_ppl={log_ppl:.3f}")

    # --- Soft Red List sweep ---
    print("\n--- Soft Red List ---")
    for delta in soft_red_list_deltas:
        print(f"\ndelta={delta}")
        generator = WatermarkedGenerator(model_name=args.model, key=args.key, top_k=args.top_k, delta=delta)
        wm_scores, wm_texts = [], []
        for i, prompt in enumerate(prompts):
            wm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True, method="soft_red_list")
            if not wm.token_ids:
                continue
            wm_scores.append(z_score(wm.token_ids, args.key))
            wm_texts.append(wm.text)
            print(f"  [{i+1}/{len(prompts)}] z_score={wm_scores[-1]:.3f}")

        tpr, threshold = tpr_at_fpr(wm_scores, uwm_green_fractions, fpr=args.fpr)
        log_ppl = mean_log_perplexity(generator.model, generator.tokenizer, wm_texts, generator.device)

        results["soft_red_list"][delta] = {
            "mean_z_score": statistics.mean(wm_scores) if wm_scores else float("nan"),
            "tpr_at_fpr": tpr,
            "mean_log_perplexity": log_ppl,
            "n_samples": len(wm_scores),
        }
        print(f"  delta={delta}: TPR@FPR={args.fpr}={tpr:.3f}  mean_log_ppl={log_ppl:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    try:
        import matplotlib.pyplot as plt

        fig_dir = Path(args.fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

        t_points = [(v["mean_log_perplexity"], v["tpr_at_fpr"]) for v in results["tournament"].values()]
        s_points = [(v["mean_log_perplexity"], v["tpr_at_fpr"]) for v in results["soft_red_list"].values()]
        t_points.sort()
        s_points.sort()

        plt.figure(figsize=(5, 4))
        if t_points:
            plt.plot([p[0] for p in t_points], [p[1] for p in t_points], marker="o",
                     label="SynthID-Text (distortionary)")
        if s_points:
            plt.plot([p[0] for p in s_points], [p[1] for p in s_points], marker="o",
                     label="Soft Red List")
        plt.xlabel("Mean log(perplexity)  [lower = better text quality]")
        plt.ylabel(f"TPR @ FPR={args.fpr}")
        plt.title("Detectability vs. quality trade-off")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        fig_path = fig_dir / "distortionary_comparison.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Figure saved to {fig_path}")
    except ImportError:
        print("matplotlib not installed -- skipping figure generation (JSON results still saved).")


if __name__ == "__main__":
    main()