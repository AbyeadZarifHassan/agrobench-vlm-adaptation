"""Convert our prediction JSONL into the official AgroBench format.

    python src/export_official.py outputs/baseline_full.jsonl \
        --out outputs/official/qwen2vl2b/all/predictions.jsonl

Then score it with the benchmark authors' own code:

    cd E:\\AgroBench-official
    python -m scripts.run_eval --pred <that path>

Running both scorers over the same predictions and getting the same accuracy
is cheap evidence that our harness implements the published protocol. Put that
line in the report -- it converts "I wrote my own evaluation" from a risk into
a verified claim.

The official scorer compares pred_letter to gold_letter as exact strings, with
no normalisation, so the mapping is just index -> letter.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def to_official(record: dict) -> dict:
    """Map one of our records onto the official row shape.

    An abstention (unparseable generation) has no letter. We emit None rather
    than inventing one: under exact-string comparison None never equals a gold
    letter, so it scores as wrong, which is the correct treatment.
    """
    pred = record.get("pred_index")
    return {
        "id": record["sample_id"],
        "category": record.get("category", "unknown"),
        "pred_letter": LETTERS[pred] if pred is not None else None,
        "gold_letter": LETTERS[record["gold_index"]],
    }


def local_score(rows: list[dict]) -> dict:
    """Reimplementation of the official compute_accuracy, used to check that
    our own numbers agree before handing the file to their scorer."""
    total = correct = 0
    by_cat_total: dict[str, int] = defaultdict(int)
    by_cat_correct: dict[str, int] = defaultdict(int)
    for r in rows:
        total += 1
        cat = r.get("category", "unknown")
        by_cat_total[cat] += 1
        if r.get("pred_letter") == r.get("gold_letter"):
            correct += 1
            by_cat_correct[cat] += 1
    return {
        "overall": correct / total if total else 0.0,
        "per_category": {
            c: by_cat_correct[c] / by_cat_total[c] if by_cat_total[c] else 0.0
            for c in by_cat_total
        },
        "n": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export to official AgroBench format")
    ap.add_argument("pred", type=Path, help="our outputs/<run>.jsonl")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination predictions.jsonl")
    args = ap.parse_args()

    records = []
    with args.pred.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    rows = [to_official(r) for r in records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    ours = sum(r["correct"] for r in records) / len(records) if records else 0.0
    theirs = local_score(rows)

    print(f"[out] {args.out}  ({len(rows)} rows)")
    print(f"  our accuracy            {ours:.6f}")
    print(f"  official-protocol score {theirs['overall']:.6f}")
    if abs(ours - theirs["overall"]) < 1e-9:
        print("  scorers agree exactly")
    else:
        print("  [!] MISMATCH -- investigate before reporting either number")

    n_abstain = sum(1 for r in rows if r["pred_letter"] is None)
    if n_abstain:
        print(f"  note: {n_abstain} abstentions exported as null (scored wrong)")

    print("\nNow run the authors' scorer on the same file:")
    print("  cd E:\\AgroBench-official")
    print(f"  python -m scripts.run_eval --pred {args.out}")


if __name__ == "__main__":
    main()
