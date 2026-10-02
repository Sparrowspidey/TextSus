"""Download ELI5 questions and split into dev/test prompt sets.

Usage:
    uv run python experiments/prepare_eli5_data.py --n_dev 200 --n_test 200

Writes data/eli5_dev.jsonl and data/eli5_test.jsonl, each line
{"prompt": "..."}. Smaller n than the paper's 10,000/10,000 -- adjust
once the pipeline is proven out and you know your compute budget.

Note: the original "eli5" and "eli5_category" datasets on the Hub use a
loading script, which recent versions of the `datasets` library (3.x+)
no longer support at all, and "eli5" is also defunct (Reddit locked down
the API it depended on). We use "sentence-transformers/eli5" instead --
a parquet-format mirror of ELI5 question/answer pairs built for training
sentence embeddings, which loads fine with no script and no login.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

# Column names to check, in priority order, across possible dataset formats.
_QUESTION_COLUMN_CANDIDATES = ["question", "anchor", "query", "title", "text"]


def _extract_questions(ds) -> list[str]:
    columns = ds.column_names
    for col in _QUESTION_COLUMN_CANDIDATES:
        if col in columns:
            print(f"Using column '{col}' as the prompt text.")
            return [str(x).strip() for x in ds[col] if x and str(x).strip()]
    raise KeyError(
        f"None of {_QUESTION_COLUMN_CANDIDATES} found in dataset columns: {columns}. "
        "Update _QUESTION_COLUMN_CANDIDATES in this script to match."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dev", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    random.seed(args.seed)

    ds = load_dataset("sentence-transformers/eli5", "pair", split="train")
    questions = _extract_questions(ds)
    questions = list(dict.fromkeys(questions))  # de-duplicate, keep order
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