"""Quality evaluation: does watermarking hurt text quality or diversity?

Generates watermarked and unwatermarked responses for a batch of prompts,
then compares:
  - perplexity (lower = more fluent; paper finds no significant difference)
  - Self-BLEU (lower = more diverse; paper finds watermarking reduces
    diversity somewhat, but less than the Gumbel baseline)

Usage:
    uv run python experiments/run_quality.py --model gpt2 --n_prompts 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textsus.evaluation.perplexity import compute_perplexity
from textsus.evaluation.self_bleu import self_bleu
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
    parser.add_argument("--n_prompts", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--out", type=str, default="results/tables/quality.json")
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

    generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=args.m)

    wm_texts, uwm_texts = [], []
    wm_perplexities, uwm_perplexities = [], []

    for i, prompt in enumerate(prompts):
        wm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True)
        uwm = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=False)

        wm_texts.append(wm.text)
        uwm_texts.append(uwm.text)

        wm_ppl = compute_perplexity(generator.model, generator.tokenizer, wm.text, generator.device)
        uwm_ppl = compute_perplexity(generator.model, generator.tokenizer, uwm.text, generator.device)
        wm_perplexities.append(wm_ppl)
        uwm_perplexities.append(uwm_ppl)

        print(f"[{i+1}/{len(prompts)}] wm_ppl={wm_ppl:.2f} uwm_ppl={uwm_ppl:.2f}")

    wm_valid = [p for p in wm_perplexities if p == p]  # drop NaNs
    uwm_valid = [p for p in uwm_perplexities if p == p]
    mean_wm_ppl = sum(wm_valid) / len(wm_valid) if wm_valid else float("nan")
    mean_uwm_ppl = sum(uwm_valid) / len(uwm_valid) if uwm_valid else float("nan")

    wm_diversity = self_bleu(wm_texts)
    uwm_diversity = self_bleu(uwm_texts)

    result = {
        "model": args.model,
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "m": args.m,
        "mean_perplexity_watermarked": mean_wm_ppl,
        "mean_perplexity_unwatermarked": mean_uwm_ppl,
        "self_bleu_watermarked": wm_diversity,
        "self_bleu_unwatermarked": uwm_diversity,
        "watermarked_texts": wm_texts,
        "unwatermarked_texts": uwm_texts,
        "watermarked_perplexities": wm_perplexities,
        "unwatermarked_perplexities": uwm_perplexities,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nMean perplexity  -- watermarked: {mean_wm_ppl:.2f}  unwatermarked: {mean_uwm_ppl:.2f}")
    print(f"Self-BLEU        -- watermarked: {wm_diversity:.2f}  unwatermarked: {uwm_diversity:.2f}")
    print(f"(Self-BLEU: higher = less diverse/more repetitive)")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()