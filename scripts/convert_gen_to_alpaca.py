#!/usr/bin/env python3
"""Convert gen.py output (outputs: list[str]) → AlpacaEval format (output: str)."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="gen.py JSON output file")
    parser.add_argument("--output", required=True, help="AlpacaEval-compatible JSON output file")
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    converted = []
    for i, row in enumerate(rows):
        outputs = row.get("outputs", row.get("output"))
        if isinstance(outputs, list):
            output = outputs[0]
        else:
            output = outputs
        converted.append({
            "instruction": row["instruction"],
            "output": output,
            "generator": row.get("generator", "unknown"),
            "dataset": "alpaca_eval_2",
            "datasplit": "eval",
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(converted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Converted {len(converted)} rows: {args.input} → {args.output}")


if __name__ == "__main__":
    main()
