"""First end-to-end experiment: does the detector actually separate
watermarked from unwatermarked text?

Usage:
    uv run python experiments/run_detectability.py \
        --model gpt2 --n_prompts 20 --max_new_tokens 100

Swap --model for google/gemma-2b-it once you have Hugging Face access;
gpt2 is a fast, no-login sanity-check default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textsus.detection.detector import Detector
from textsus.evaluation.metrics import tpr_at_fpr
from textsus.generation.watermarked_generator import WatermarkedGenerator


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
    parser.add_argument("--n_prompts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--m", type=int, default=4, help="tournament layers")
    parser.add_argument("--H", type=int, default=4, help="seed sliding-window size")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="results/tables/detectability.json")
    args = parser.parse_args()

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

    generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=args.m, H=args.H)
    detector = Detector(key=args.key, m=args.m, H=args.H)

    watermarked_scores, unwatermarked_scores = [], []

    for i, prompt in enumerate(prompts):
        wm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True)
        uwm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=False)

        watermarked_scores.append(detector.score(wm.token_ids))
        unwatermarked_scores.append(detector.score(uwm.token_ids))
        print(f"[{i+1}/{len(prompts)}] wm_score={watermarked_scores[-1]:.3f} "
              f"uwm_score={unwatermarked_scores[-1]:.3f}")

    tpr, threshold = tpr_at_fpr(watermarked_scores, unwatermarked_scores, fpr=args.fpr)

    result = {
        "model": args.model,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "m": args.m,
        "H": args.H,
        "fpr": args.fpr,
        "threshold": threshold,
        "tpr_at_fpr": tpr,
        "watermarked_scores": watermarked_scores,
        "unwatermarked_scores": unwatermarked_scores,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nTPR @ FPR={args.fpr}: {tpr:.3f} (threshold={threshold:.3f})")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()