"""Run a VLM over AgroBench and write per-sample predictions to JSONL.

    python src/evaluate.py --run-name baseline --limit 200
    python src/evaluate.py --run-name baseline_full --categories disease_identification

Predictions stream to disk one line at a time and the run resumes from
whatever is already there, so an interrupted 1,500-sample run costs you
nothing. Scoring lives in analyze.py: this script only produces raw
predictions, which means you can re-score without re-running the model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data import iter_samples  # noqa: E402

DATASET_ID = "risashinoda/AgroBench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a VLM on AgroBench")
    p.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--adapter-path", default=None,
                   help="LoRA adapter dir; omit for the baseline run")
    p.add_argument("--run-name", required=True,
                   help="Output goes to outputs/<run-name>.jsonl")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--decode", choices=["logit", "generate"], default="logit")
    p.add_argument("--prompt-style", choices=["official", "strict"],
                   default="official",
                   help="official reproduces the authors' prompt verbatim")
    p.add_argument("--parser", choices=["official", "strict"], default="official",
                   help="generate-mode answer parser; official mirrors theirs")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Filter by the category field")
    p.add_argument("--sources", nargs="*", default=None,
                   help="Filter by the source field. On AgroBench this is the "
                        "task identifier (e.g. did for disease identification)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28,
                   help="Vision token budget. Must match across compared runs.")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="Permute options to measure position bias")
    p.add_argument("--shuffle-images", type=int, default=None,
                   help="CONTROL: pair each question with another sample's "
                        "image (seeded derangement). A model genuinely using "
                        "the image should collapse toward chance. If accuracy "
                        "survives, it learned text priors, not vision.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--split", default="train")
    p.add_argument("--local-jsonl", type=Path, default=None,
                   help="evaluate a locally built MCQ file (see build_mcq.py) "
                        "instead of the gated Hub dataset")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list-categories", action="store_true",
                   help="Print category counts and exit (no model loaded)")
    return p.parse_args()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a torn final line from a hard kill
    return done


def run_config(args, extra: dict) -> dict:
    """Everything needed to reproduce or invalidate this run.

    If two runs you are comparing differ in any field here other than
    model_id/adapter_path, the comparison is not clean.
    """
    cfg = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "decode": args.decode,
        "prompt_style": args.prompt_style,
        "parser": args.parser,
        "max_pixels": args.max_pixels,
        "shuffle_seed": args.shuffle_seed,
        "shuffle_images": args.shuffle_images,
        "seed": args.seed,
        "quantized_4bit": not args.no_4bit,
        "dataset_id": (str(args.local_jsonl) if args.local_jsonl
                       else args.dataset_id),
        "split": args.split,
        "categories": args.categories,
        "sources": args.sources,
        "limit": args.limit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cfg.update(extra)
    return cfg


def load_local_jsonl(path: Path) -> list[dict]:
    """Read locally built MCQ records and lazily attach images.

    Images are opened on access rather than all at once, so a 5,000-question
    set does not need to fit in RAM.
    """
    from PIL import Image

    class LazyRow(dict):
        def __getitem__(self, key):
            if key == "image" and "image" not in self:
                return Image.open(super().__getitem__("image_path"))
            return super().__getitem__(key)

        def get(self, key, default=None):
            try:
                return self[key]
            except KeyError:
                return default

    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(LazyRow(json.loads(line)))
    return rows


def derange_images(rows: list, seed: int) -> list:
    """Reassign every question a different sample's image.

    This is the control that separates "the model reads the image" from "the
    model learned which answer tends to be correct". A model using visual
    evidence should fall to near chance here. One that holds its accuracy has
    learned the text distribution, and its headline number means nothing.

    A derangement is used rather than a plain shuffle, so no sample keeps its
    own image by luck. Preference is given to donors of a DIFFERENT class, so
    a question about late blight does not accidentally receive another late
    blight image and score correctly for the right reason.
    """
    import random as _random

    n = len(rows)
    if n < 2:
        return rows

    rng = _random.Random(seed)
    order = list(range(n))
    labels = [r["answer"] for r in rows]
    paths = [r["image_path"] for r in rows]

    for _ in range(200):
        rng.shuffle(order)
        if all(order[i] != i for i in range(n)):
            break
    else:  # fall back to a rotation, which is always a derangement
        order = [(i + 1) % n for i in range(n)]

    # Try to repair same-class assignments by swapping with a later index.
    for i in range(n):
        if labels[order[i]] == labels[i]:
            for j in range(n):
                if (i != j and labels[order[j]] != labels[i]
                        and labels[order[i]] != labels[j]
                        and order[j] != i and order[i] != j):
                    order[i], order[j] = order[j], order[i]
                    break

    same = sum(1 for i in range(n) if labels[order[i]] == labels[i])
    print(f"[control] {same}/{n} ({same/n:.1%}) deranged images still share the "
          "gold class; those can be correct by chance")

    for i, row in enumerate(rows):
        row["image_path"] = paths[order[i]]
        row.pop("image", None)
    return rows


def main() -> None:
    args = parse_args()

    if args.local_jsonl:
        print(f"[info] loading local MCQ file {args.local_jsonl}")
        ds = load_local_jsonl(args.local_jsonl)
        print(f"[info] {len(ds)} rows")
    else:
        from datasets import load_dataset

        print(f"[info] loading {args.dataset_id} ({args.split})")
        ds = load_dataset(args.dataset_id, split=args.split)
        print(f"[info] {len(ds)} rows")

    if args.shuffle_images is not None:
        if not args.local_jsonl:
            raise SystemExit("--shuffle-images requires --local-jsonl")
        ds = derange_images(ds, args.shuffle_images)
        print(f"[control] images deranged with seed {args.shuffle_images}; "
              "no sample keeps its own image")

    if args.list_categories:
        from collections import Counter
        for field in ("source", "category"):
            try:
                counts = Counter(str(r.get(field, "")) for r in ds)
            except Exception:
                continue
            if not counts or list(counts) == [""]:
                continue
            print(f"\n=== {field} ({len(counts)} distinct) ===")
            width = max(len(c) for c in counts)
            for name, n in counts.most_common(40):
                print(f"  {name:<{width}}  {n:>5}")
            if len(counts) > 40:
                print(f"  ... and {len(counts)-40} more")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{args.run_name}.jsonl"
    cfg_path = outdir / f"{args.run_name}.config.json"

    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = load_done_ids(out_path)
    if done:
        print(f"[info] resuming, {len(done)} samples already done")

    import torch
    from model import load_model, score_by_generation, score_by_letter_logits, set_seed

    set_seed(args.seed)
    print(f"[info] loading {args.model_id}"
          + (f" + adapter {args.adapter_path}" if args.adapter_path else ""))
    lm = load_model(
        model_id=args.model_id,
        load_in_4bit=not args.no_4bit,
        max_pixels=args.max_pixels,
        adapter_path=args.adapter_path,
    )

    env = {
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    cfg_path.write_text(json.dumps(run_config(args, env), indent=2),
                        encoding="utf-8")

    if args.decode == "logit":
        def scorer(lm_, img, prompt, n_opts, opts):
            return score_by_letter_logits(lm_, img, prompt, n_opts)
    else:
        def scorer(lm_, img, prompt, n_opts, opts):
            return score_by_generation(
                lm_, img, prompt, n_opts, options=opts, parser=args.parser
            )

    n_done = n_correct = 0
    start = time.time()

    with out_path.open("a", encoding="utf-8") as fh:
        for sample in iter_samples(
            ds,
            categories=args.categories,
            sources=args.sources,
            limit=args.limit,
            shuffle_seed=args.shuffle_seed,
            prompt_style=args.prompt_style,
        ):
            if sample.sample_id in done:
                continue
            try:
                result = scorer(
                    lm, sample.image, sample.prompt,
                    len(sample.options), sample.options,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[warn] OOM on {sample.sample_id}; lower --max-pixels")
                raise

            pred = result["pred_index"]
            correct = (pred is not None) and (pred == sample.gold_index)
            record = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "source": sample.source,
                "crop": sample.crop,
                "question": sample.question,
                "options": sample.options,
                "gold_index": sample.gold_index,
                "gold_text": sample.gold_text,
                "pred_index": pred,
                "pred_text": sample.options[pred] if pred is not None else None,
                "correct": bool(correct),
                "abstained": pred is None,
                "confidence": result["confidence"],
                "letter_scores": result["letter_scores"],
                "raw_output": result["raw_output"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()  # survive a kill mid-run

            n_done += 1
            n_correct += int(correct)
            if n_done % 25 == 0:
                rate = n_done / (time.time() - start)
                print(f"  {n_done} done | running acc {n_correct/n_done:.3f} "
                      f"| {rate:.2f} it/s")

    if n_done:
        elapsed = time.time() - start
        print(f"\n[done] {n_done} new samples in {elapsed/60:.1f} min "
              f"| accuracy {n_correct/n_done:.4f}")
    else:
        print("\n[done] nothing new to evaluate")
    print(f"[out]  {out_path}")
    print(f"[next] python src/analyze.py {out_path}")


if __name__ == "__main__":
    main()
