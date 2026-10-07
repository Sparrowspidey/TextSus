"""Interactive demo: generate watermarked text on demand, and check whether
ANY text you paste in (watermarked output, edited text, or genuine human
writing) is detected as watermarked. Good for a live walkthrough.

Usage:
    uv run python experiments/interactive_demo.py --model gpt2
    uv run python experiments/interactive_demo.py --model google/gemma-2b-it
"""

from __future__ import annotations

import argparse

from textsus.generation.watermarked_generator import WatermarkedGenerator
from textsus.scoring.mean_score import mean_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    # Rough threshold; the real one should come from tpr_at_fpr() on a real
    # unwatermarked sample, but ~0.55-0.58 is a reasonable live-demo default
    # based on this project's earlier detectability runs.
    parser.add_argument("--threshold", type=float, default=0.56)
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    generator = WatermarkedGenerator(model_name=args.model, key=args.key, m=args.m)
    print("Ready.\n")

    while True:
        print("=" * 60)
        print("1) Generate watermarked text from a prompt")
        print("2) Check whether some text is watermarked (paste anything)")
        print("3) Quit")
        choice = input("> ").strip()

        if choice == "1":
            prompt = input("Prompt: ").strip()
            result = generator.generate(prompt, max_new_tokens=args.max_new_tokens, watermark=True)
            print("\n--- Watermarked output ---")
            print(result.text)
            score = mean_score(result.token_ids, args.key, num_layers=args.m)
            print(f"\nDetection score: {score:.3f}  (threshold: {args.threshold})")
            print("Verdict: WATERMARKED" if score > args.threshold else "Verdict: NOT watermarked")

        elif choice == "2":
            print("Paste the text to check, then press Enter twice:")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            text = "\n".join(lines)
            if not text.strip():
                print("(empty input, skipping)\n")
                continue

            token_ids = generator.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
            score = mean_score(token_ids, args.key, num_layers=args.m)
            print(f"\nDetection score: {score:.3f}  (threshold: {args.threshold})")
            print("Verdict: WATERMARKED" if score > args.threshold else "Verdict: NOT watermarked")
            print(f"({len(token_ids)} tokens scored)\n")

        elif choice == "3":
            break
        else:
            print("Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()