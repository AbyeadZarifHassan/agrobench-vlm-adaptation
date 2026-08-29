"""Convert an image-classification dataset into AgroBench-style MCQ records.

AgroBench is gated. This adapter lets the same evaluation harness run on
ungated agricultural datasets (PlantVillage, PlantDoc, IP102) by turning
"image + class label" into "image + question + options + answer", using the
exact record shape data.py already expects.

    # inspect what a folder contains
    python src/build_mcq.py --path data/PlantVillage --inspect

    # build eval sets
    python src/build_mcq.py --path data/PlantVillage --out data/pv_test.jsonl \
        --split test --split-frac 0.15
    python src/build_mcq.py --path data/PlantDoc --out data/plantdoc_all.jsonl

DESIGN NOTES

Distractor choice determines how hard the benchmark is, so it is not random.
Distractors are drawn preferentially from the SAME CROP as the gold label
("Tomato late blight" vs "Tomato early blight" rather than vs "Corn rust"),
which mirrors how AgroBench builds its options and keeps the task about
fine-grained disease discrimination rather than crop recognition. A benchmark
built with cross-crop distractors is trivially easy and any improvement on it
is uninteresting.

Everything is seeded on the sample's own path, so the generated benchmark is
byte-identical across machines and across reruns. Regenerating with a
different seed after seeing your results would be a way to fool yourself.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

QUESTION_DISEASE = "What is the disease of the plant in this image?"
QUESTION_HEALTHY = "What is the condition of the plant in this image?"


# --------------------------------------------------------------------------
# label parsing
# --------------------------------------------------------------------------

# Crop names some releases write inverted or with a parenthetical.
CROP_FIXES = {
    "pepper bell": "Bell pepper",
    "pepper, bell": "Bell pepper",
    "corn maize": "Corn",
}


def _split_camel(s: str) -> str:
    """YellowLeaf -> Yellow Leaf. PlantVillage folder names mix camel case
    into otherwise underscore-separated labels."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)


def parse_label(raw: str) -> tuple[str, str, str]:
    """Split a folder name into (crop, condition, display).

    PlantVillage releases are inconsistent about separators, using three, two
    or one underscore between crop and condition, sometimes within a single
    directory listing. The separator is resolved longest-first, because the
    longest one present is the real crop/condition boundary:

      Pepper__bell___Bacterial_spot    -> ("Bell pepper", "Bacterial spot")
      Tomato__Tomato_YellowLeaf__Curl_Virus
                                       -> ("Tomato", "Yellow leaf curl virus")
      Tomato_Bacterial_spot            -> ("Tomato", "Bacterial spot")
      Potato___healthy                 -> ("Potato", "healthy")
      Corn_(maize)___Common_rust_      -> ("Corn", "Common rust")
    """
    name = re.sub(r"\(.*?\)", "", str(raw).strip())      # drop "(maize)"

    for sep in ("___", "__", "_"):
        if sep in name:
            crop_part, _, cond_part = name.partition(sep)
            break
    else:
        crop_part, cond_part = name, ""

    def clean(s: str) -> str:
        s = re.sub(r"[_]+", " ", s)
        s = _split_camel(s)
        return re.sub(r"\s+", " ", s).strip(" ,")

    crop = clean(crop_part)
    crop = CROP_FIXES.get(crop.lower(), crop).title()
    # .title() would turn "Bell pepper" into "Bell Pepper"; keep the fix's form
    if crop.lower() in {v.lower() for v in CROP_FIXES.values()}:
        crop = next(v for v in CROP_FIXES.values() if v.lower() == crop.lower())

    cond = clean(cond_part)
    # Some labels repeat the crop inside the condition ("Tomato_Tomato_mosaic")
    cond = re.sub(rf"^{re.escape(crop)}\s+", "", cond, flags=re.I).strip()
    if not cond:
        cond = "healthy"

    if cond.lower() in ("healthy", "health"):
        cond = "healthy"
        display = f"Healthy {crop.lower()}"
    else:
        display = f"{crop} {cond.lower()}"
        cond = cond[0].upper() + cond[1:]

    return crop, cond, display


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def _norm_key(s: str) -> str:
    """Normalise a folder name for label-map lookup.

    Datasets are inconsistent about case, spaces and underscores for what is
    the same class ("Tomato_Early_blight_leaf", "Tomato Early Blight Leaf").
    Requiring an exact match would silently skip every folder and produce an
    empty eval set, so keys are compared in normalised form.
    """
    return re.sub(r"[\s_]+", " ", str(s)).strip().lower()


def scan_imagefolder(root: Path, label_map: dict | None = None) -> list[dict]:
    """Collect (path, label) pairs from a class-per-subfolder layout.

    label_map, when given, maps a raw folder name to a canonical display label.
    This matters for cross-dataset transfer testing: the evaluation set must
    use the SAME answer strings the model was trained to produce, or a drop in
    accuracy measures vocabulary mismatch rather than domain shift. Folders
    absent from the map are skipped and reported, so classes with no
    counterpart in training are excluded rather than silently scored wrong.
    """
    items: list[dict] = []
    skipped: Counter = Counter()

    norm_map = None
    if label_map is not None:
        norm_map = {_norm_key(k): v for k, v in label_map.items()}

    for class_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        images = [p for p in sorted(class_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        if norm_map is not None:
            mapped = norm_map.get(_norm_key(class_dir.name))
            if mapped is None:
                skipped[class_dir.name] = len(images)
                continue
            display = mapped
            crop = display.replace("Healthy ", "").split()[0].title()
            if "bell pepper" in display.lower():
                crop = "Bell pepper"
            cond = "healthy" if display.lower().startswith("healthy") else display
        else:
            crop, cond, display = parse_label(class_dir.name)

        for img in images:
            items.append({
                "path": str(img.resolve()),
                "raw_label": class_dir.name,
                "crop": crop,
                "condition": cond,
                "label": display,
            })

    if skipped:
        print(f"[skip] {len(skipped)} folders not in the label map "
              f"({sum(skipped.values())} images):")
        for name, n in skipped.most_common():
            print(f"    {name}  ({n})")

    return items


# --------------------------------------------------------------------------
# MCQ construction
# --------------------------------------------------------------------------

def build_options(
    gold: str,
    crop: str,
    by_crop: dict[str, list[str]],
    all_labels: list[str],
    n_options: int,
    rng: random.Random,
) -> list[str]:
    """Pick n_options-1 distractors, preferring same-crop labels."""
    pool = [l for l in by_crop.get(crop, []) if l != gold]
    rng.shuffle(pool)
    chosen = pool[: n_options - 1]

    if len(chosen) < n_options - 1:
        extra = [l for l in all_labels if l != gold and l not in chosen]
        rng.shuffle(extra)
        chosen += extra[: n_options - 1 - len(chosen)]

    options = chosen + [gold]
    rng.shuffle(options)
    return options


def build_records(
    items: list[dict],
    n_options: int,
    seed: int,
    max_per_class: int | None,
    split: str,
    split_frac: float,
    extra_labels: list[str] | None = None,
) -> list[dict]:
    by_crop: dict[str, list[str]] = defaultdict(set)
    for it in items:
        by_crop[it["crop"]].add(it["label"])

    # When an external vocabulary is supplied, distractors are drawn from it
    # rather than only from labels present in this dataset. Without this, a
    # transfer set with fewer classes would have an easier option space than
    # the training set, and the accuracy drop would be understated.
    if extra_labels:
        for lab in extra_labels:
            crop = lab.replace("Healthy ", "").split()[0].title()
            if "bell pepper" in lab.lower():
                crop = "Bell pepper"
            by_crop[crop].add(lab)

    by_crop = {k: sorted(v) for k, v in by_crop.items()}
    all_labels = sorted(set(extra_labels or []) |
                        {it["label"] for it in items})

    # Deterministic per-class split. Hashing the path means an image lands in
    # the same split no matter the scan order or the machine -- essential when
    # one split trains the model and the other evaluates it.
    def in_eval(path: str) -> bool:
        h = random.Random(f"split:{seed}:{path}").random()
        return h < split_frac

    per_class: Counter = Counter()
    records: list[dict] = []

    for it in sorted(items, key=lambda x: x["path"]):
        if split == "test" and not in_eval(it["path"]):
            continue
        if split == "train" and in_eval(it["path"]):
            continue
        if max_per_class is not None and per_class[it["label"]] >= max_per_class:
            continue
        per_class[it["label"]] += 1

        rng = random.Random(f"opts:{seed}:{it['path']}")
        options = build_options(
            it["label"], it["crop"], by_crop, all_labels, n_options, rng
        )
        question = (QUESTION_HEALTHY if it["condition"] == "healthy"
                    else QUESTION_DISEASE)

        records.append({
            "id": _stable_id(it["path"]),
            "image_path": it["path"],
            "question": question,
            "options": options,
            "answer": it["label"],
            "category": "disease_identification",
            "crop": it["crop"],
            "source": Path(it["path"]).parent.parent.name,
        })

    return records


def _stable_id(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}__{p.stem}".replace(" ", "_")[:120]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def inspect(items: list[dict]) -> None:
    by_label = Counter(it["label"] for it in items)
    by_crop = Counter(it["crop"] for it in items)
    print(f"{len(items)} images, {len(by_label)} classes, {len(by_crop)} crops\n")
    print("CROPS")
    for crop, n in by_crop.most_common():
        labels = sorted({it["label"] for it in items if it["crop"] == crop})
        print(f"  {crop:<16} {n:>6} images  {len(labels)} classes")
        for l in labels:
            print(f"      {l}  ({by_label[l]})")
    thin = [l for l, n in by_label.items() if n < 20]
    if thin:
        print(f"\n[note] {len(thin)} classes have under 20 images; "
              "their per-class accuracy will be very noisy")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build AgroBench-style MCQ records from a classification dataset"
    )
    ap.add_argument("--path", type=Path, required=True,
                    help="dataset root with one folder per class")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inspect", action="store_true",
                    help="print class structure and exit")
    ap.add_argument("--n-options", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap per class to keep the eval set balanced and fast")
    ap.add_argument("--split", choices=["all", "train", "test"], default="all")
    ap.add_argument("--split-frac", type=float, default=0.15,
                    help="fraction assigned to test when --split is used")
    ap.add_argument("--label-map", type=Path, default=None,
                    help="JSON mapping raw folder name -> canonical label. "
                         "Required for cross-dataset transfer tests so the "
                         "answer vocabulary matches the training set.")
    ap.add_argument("--option-vocab", type=Path, default=None,
                    help="JSONL or JSON list of labels to draw distractors "
                         "from. Use the TRAINING set's label list so the "
                         "option space is identical across datasets.")
    args = ap.parse_args()

    if not args.path.exists():
        raise SystemExit(f"no such directory: {args.path}")

    label_map = None
    if args.label_map:
        label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
        print(f"[map] {len(label_map)} folder names mapped")

    items = scan_imagefolder(args.path, label_map)
    if not items:
        raise SystemExit(
            f"no images found under {args.path}. Expected one folder per class."
        )

    if args.inspect:
        inspect(items)
        return

    if args.out is None:
        raise SystemExit("--out is required unless --inspect is given")

    extra_labels = None
    if args.option_vocab:
        raw = args.option_vocab.read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            extra_labels = json.loads(raw)
        else:  # a jsonl of MCQ records -- collect their answers
            extra_labels = sorted({
                json.loads(l)["answer"] for l in raw.splitlines() if l.strip()
            })
        print(f"[vocab] drawing distractors from {len(extra_labels)} "
              "labels supplied externally")

    records = build_records(
        items, args.n_options, args.seed, args.max_per_class,
        args.split, args.split_frac, extra_labels,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = Counter(r["answer"] for r in records)
    print(f"[out] {args.out}")
    print(f"  {len(records)} questions, {len(labels)} classes, "
          f"{args.n_options} options each")
    print(f"  chance accuracy = {1/args.n_options:.3f}")

    same_crop = sum(
        1 for r in records
        if all(o.split()[0].lower() == r["crop"].lower().split()[0]
               or o.lower().startswith("healthy")
               for o in r["options"])
    )
    print(f"  {same_crop}/{len(records)} ({same_crop/len(records):.0%}) have "
          "all distractors from the same crop")
    print(f"\n[next] python src/evaluate.py --run-name baseline "
          f"--local-jsonl {args.out}")


if __name__ == "__main__":
    main()
