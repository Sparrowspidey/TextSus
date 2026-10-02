"""Download ELI5 questions and split into dev/test prompt sets.

Usage:
    uv run python experiments/prepare_eli5_data.py --n_dev 200 --n_test 200

Writes data/eli5_dev.jsonl and data/eli5_test.jsonl, each line
{"prompt": "..."}. Smaller n than the paper's 10,000/10,000 -- adjust
once the pipeline is proven out and you know your compute budget.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dev", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)

    # eli5_category is the maintained replacement for the original eli5 dataset.
    ds = load_dataset("eli5_category", split="train")
    questions = [ex["title"].strip() for ex in ds if ex.get("title")]
    random.shuffle(questions)

    needed = args.n_dev + args.n_test
    if len(questions) < needed:
        raise ValueError(f"Only {len(questions)} questions available, need {needed}.")

    dev = questions[: args.n_dev]
    test = questions[args.n_dev : args.n_dev + args.n_test]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in [("eli5_dev.jsonl", dev), ("eli5_test.jsonl", test)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for q in split:
                f.write(json.dumps({"prompt": q}) + "\n")

    print(f"Wrote {len(dev)} dev and {len(test)} test prompts to {out_dir}/")


if __name__ == "__main__":
    main()