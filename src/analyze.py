"""Score AgroBench prediction files and diagnose where a model fails.

    python src/analyze.py outputs/baseline.jsonl
    python src/analyze.py outputs/baseline.jsonl --compare outputs/lora_v1.jsonl

Single-run mode produces the error analysis that should decide WHICH
modification to make. Compare mode produces the paired significance test that
tells you whether an improvement is real.

No torch, no GPU -- run this on any machine.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. Wilson rather than normal-approximation
    because per-class counts here are small and the normal interval goes
    nonsensical (below 0, above 1) exactly where the interesting classes are."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar's test.

    b = baseline wrong, modified right (your gains)
    c = baseline right, modified wrong (your regressions)
    Under the null the modification changes nothing, so each discordant pair
    is a fair coin. Exact rather than chi-square: with a few dozen discordant
    pairs the chi-square approximation is not trustworthy.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_diff_ci(
    pairs: list[tuple[bool, bool]], n_boot: int = 10000, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for (accuracy_b - accuracy_a), resampling SAMPLES not
    predictions, so the paired structure is preserved."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(pairs[i][0] for i in idx)
        b = sum(pairs[i][1] for i in idx)
        diffs.append((b - a) / n)
    diffs.sort()
    return (diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)])


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load_run(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[warn] skipping malformed line in {path.name}")

    ids = [r.get("sample_id") for r in records]
    if len(set(ids)) != len(ids):
        dupes = len(ids) - len(set(ids))
        print(f"[ERROR] {path.name} contains {dupes} duplicate sample_ids. "
              "Paired comparison keys on sample_id, so duplicates overwrite "
              "each other and the result is computed on an arbitrary subset. "
              "Re-run this evaluation with --overwrite using a version of "
              "data.py that builds unique ids.")
    return records


def _group_accuracy(records: list[dict], key: str) -> list[tuple]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r.get(key, "unknown")].append(r)
    rows = []
    for name, rs in buckets.items():
        n = len(rs)
        k = sum(r["correct"] for r in rs)
        lo, hi = wilson_interval(k, n)
        chance = sum(1 / max(1, len(r["options"])) for r in rs) / n
        rows.append((name, n, k / n, lo, hi, chance))
    return sorted(rows, key=lambda r: r[2])


# --------------------------------------------------------------------------
# single-run report
# --------------------------------------------------------------------------

def report_single(records: list[dict], top_k: int = 12) -> dict:
    n = len(records)
    if n == 0:
        print("no records")
        return {}
    k = sum(r["correct"] for r in records)
    acc = k / n
    lo, hi = wilson_interval(k, n)
    chance = sum(1 / max(1, len(r["options"])) for r in records) / n
    abstained = sum(r.get("abstained", False) for r in records)

    print("=" * 72)
    print(f"OVERALL   n={n}   accuracy={acc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"          chance={chance:.4f}   above chance by {acc - chance:+.4f}")
    if abstained:
        print(f"          abstentions (unparseable) = {abstained} "
              f"({abstained/n:.1%}) -- these count as wrong")
    print("=" * 72)

    # ---- per task ----
    # AgroBench's category field is a fine-grained taxonomy with hundreds of
    # values, most holding one or two samples. A row with n=1 reads as 0.000 or
    # 1.000 accuracy and means nothing, so only groups large enough to
    # interpret are shown, with the remainder pooled.
    MIN_N = 20
    cat_rows = _group_accuracy(records, "category")
    big = [r for r in cat_rows if r[1] >= MIN_N]
    small = [r for r in cat_rows if r[1] < MIN_N]

    print(f"\nPER CATEGORY  (worst first; groups with n<{MIN_N} pooled below)")
    print(f"  {'category':<34} {'n':>5} {'acc':>7} {'95% CI':>16} {'chance':>7}")
    for name, cn, cacc, clo, chi, cch in big:
        flag = "  <-- at/below chance" if chi < cch else ""
        print(f"  {name:<34} {cn:>5} {cacc:>7.3f} "
              f"[{clo:.3f},{chi:.3f}] {cch:>7.3f}{flag}")
    if small:
        s_n = sum(r[1] for r in small)
        s_correct = sum(
            r["correct"] for r in records
            if r.get("category") in {x[0] for x in small}
        )
        s_lo, s_hi = wilson_interval(s_correct, s_n)
        print(f"  {f'[{len(small)} small groups pooled]':<34} {s_n:>5} "
              f"{s_correct/s_n:>7.3f} [{s_lo:.3f},{s_hi:.3f}]")

    # ---- per source (the AgroBench task identifier) ----
    sources = {r.get("source", "") for r in records}
    if len(sources) > 1 and sources != {""}:
        print("\nPER SOURCE / TASK  (worst first)")
        print(f"  {'source':<34} {'n':>5} {'acc':>7} {'95% CI':>16} {'chance':>7}")
        for name, cn, cacc, clo, chi, cch in _group_accuracy(records, "source"):
            flag = "  <-- at/below chance" if chi < cch else ""
            print(f"  {name:<34} {cn:>5} {cacc:>7.3f} "
                  f"[{clo:.3f},{chi:.3f}] {cch:>7.3f}{flag}")

    # ---- position bias ----
    preds = [r["pred_index"] for r in records if r["pred_index"] is not None]
    golds = [r["gold_index"] for r in records]
    pred_dist = Counter(preds)
    gold_dist = Counter(golds)
    print("\nPOSITION BIAS  (predicted letter vs gold letter)")
    for i in sorted(set(pred_dist) | set(gold_dist)):
        pp = pred_dist.get(i, 0) / max(1, len(preds))
        gp = gold_dist.get(i, 0) / max(1, len(golds))
        bar = "#" * int(pp * 40)
        print(f"  {LETTERS[i]}  pred {pp:>6.1%}  gold {gp:>6.1%}  {bar}")
    if preds:
        top_letter, top_n = pred_dist.most_common(1)[0]
        share = top_n / len(preds)
        if share > 0.45:
            print(f"  [!] {share:.0%} of predictions are '{LETTERS[top_letter]}'. "
                  "Re-run with --shuffle-seed to check how much accuracy survives.")

    # ---- confidence calibration ----
    conf = [(r["confidence"], r["correct"]) for r in records
            if r.get("confidence") is not None]
    if conf:
        print("\nCALIBRATION  (confidently wrong => knowledge gap; "
              "uniformly unsure => perception gap)")
        edges = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.01]
        for a, b in zip(edges, edges[1:]):
            bucket = [c for c in conf if a <= c[0] < b]
            if bucket:
                bacc = sum(c[1] for c in bucket) / len(bucket)
                print(f"  conf [{a:.2f},{b:.2f})  n={len(bucket):>5}  acc={bacc:.3f}")

    # ---- long tail ----
    per_class = defaultdict(lambda: [0, 0])
    for r in records:
        per_class[r["gold_text"]][1] += 1
        per_class[r["gold_text"]][0] += int(r["correct"])
    supports = sorted(v[1] for v in per_class.values())
    if supports:
        median = supports[len(supports) // 2]
        head = [(c, v) for c, v in per_class.items() if v[1] > median]
        tail = [(c, v) for c, v in per_class.items() if v[1] <= median]
        def agg(group):
            kk = sum(v[0] for _, v in group)
            nn = sum(v[1] for _, v in group)
            return kk / nn if nn else 0.0, nn
        h_acc, h_n = agg(head)
        t_acc, t_n = agg(tail)
        print(f"\nLONG TAIL  ({len(per_class)} distinct gold classes, "
              f"median support {median})")
        print(f"  head (support > {median}):  n={h_n:>5}  acc={h_acc:.3f}")
        print(f"  tail (support <= {median}): n={t_n:>5}  acc={t_acc:.3f}")
        if h_acc - t_acc > 0.10:
            print("  [!] Large head/tail gap -> class-balanced adaptation is "
                  "a well-motivated modification.")

    # ---- confusions ----
    conf_pairs = Counter(
        (r["gold_text"], r["pred_text"]) for r in records
        if not r["correct"] and r["pred_text"] is not None
    )
    print(f"\nTOP CONFUSIONS  (gold -> predicted)")
    print("  Repeated, specific pairs mean fine-grained perception failure.")
    print("  Scattered one-off errors mean a knowledge gap instead.")
    for (gold, pred), cnt in conf_pairs.most_common(top_k):
        print(f"  {cnt:>4}x  {gold[:34]:<34} -> {pred[:34]}")
    if conf_pairs:
        repeated = sum(c for c in conf_pairs.values() if c > 1)
        total_err = sum(conf_pairs.values())
        print(f"\n  {repeated}/{total_err} ({repeated/total_err:.0%}) of errors fall "
              "into repeated confusion pairs.")

    return {
        "n": n, "accuracy": acc, "ci": [lo, hi], "chance": chance,
        "abstentions": abstained,
        "per_category": {r[0]: {"n": r[1], "acc": r[2]}
                         for r in _group_accuracy(records, "category")},
    }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def report_compare(base: list[dict], mod: list[dict],
                   name_a: str, name_b: str) -> dict:
    """Paired comparison on the intersection of sample ids.

    Comparing two different subsets is the most common way to accidentally
    manufacture an improvement, so this aligns on sample_id and reports how
    many samples were dropped.
    """
    a = {r["sample_id"]: r for r in base}
    b = {r["sample_id"]: r for r in mod}
    shared = sorted(set(a) & set(b))
    if not shared:
        print("[error] no overlapping sample_ids -- these runs are not comparable")
        return {}
    if len(shared) < max(len(a), len(b)):
        print(f"[warn] comparing {len(shared)} shared samples "
              f"({name_a}: {len(a)}, {name_b}: {len(b)})")

    pairs = [(a[s]["correct"], b[s]["correct"]) for s in shared]
    n = len(pairs)
    acc_a = sum(p[0] for p in pairs) / n
    acc_b = sum(p[1] for p in pairs) / n

    both = sum(1 for x, y in pairs if x and y)
    gained = sum(1 for x, y in pairs if not x and y)
    lost = sum(1 for x, y in pairs if x and not y)
    neither = sum(1 for x, y in pairs if not x and not y)

    p_value = binom_two_sided_p(gained, lost)
    ci_lo, ci_hi = bootstrap_diff_ci(pairs)

    print("=" * 72)
    print(f"PAIRED COMPARISON   n={n} shared samples")
    print("=" * 72)
    print(f"  {name_a:<28} accuracy = {acc_a:.4f}")
    print(f"  {name_b:<28} accuracy = {acc_b:.4f}")
    print(f"  difference                   = {acc_b - acc_a:+.4f}  "
          f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"\n  CONTINGENCY")
    print(f"    both correct        {both:>5}")
    print(f"    fixed by {name_b:<12} {gained:>5}   <- gains")
    print(f"    broken by {name_b:<11} {lost:>5}   <- regressions")
    print(f"    both wrong          {neither:>5}")
    print(f"\n  McNemar exact p = {p_value:.4g}")

    if p_value < 0.05 and acc_b > acc_a:
        print("  => Improvement is statistically significant.")
    elif p_value < 0.05:
        print("  => Significant DEGRADATION. Report this; do not bury it.")
    else:
        print("  => Not significant. The difference is consistent with noise;")
        print("     say so plainly rather than reporting the raw delta as a win.")
    if ci_lo < 0 < ci_hi:
        print("  Note: the CI spans zero, which agrees with the test above.")

    # per source (AgroBench task id) and per category, small groups pooled
    for field, label in (("source", "SOURCE / TASK"), ("category", "CATEGORY")):
        vals = {a[s].get(field, "") for s in shared}
        if len(vals) <= 1 or vals == {""}:
            continue
        min_n = 20 if field == "category" else 1
        rows_out = []
        pooled_ids: list[str] = []
        for val in sorted(vals):
            ids = [s for s in shared if a[s].get(field) == val]
            if not ids:
                continue
            if len(ids) < min_n:
                pooled_ids += ids
                continue
            ca = sum(a[s]["correct"] for s in ids) / len(ids)
            cb = sum(b[s]["correct"] for s in ids) / len(ids)
            gain = sum(1 for s in ids if not a[s]["correct"] and b[s]["correct"])
            loss = sum(1 for s in ids if a[s]["correct"] and not b[s]["correct"])
            rows_out.append((val, len(ids), ca, cb, cb - ca,
                             binom_two_sided_p(gain, loss)))

        if not rows_out and not pooled_ids:
            continue
        print(f"\n  PER {label}  (sorted by delta)")
        print(f"    {'name':<28} {'n':>5} {'before':>8} {'after':>8} "
              f"{'delta':>8} {'p':>10}")
        for val, cn, ca, cb, d, pv in sorted(rows_out, key=lambda r: -r[4]):
            star = " *" if pv < 0.05 else ""
            print(f"    {val:<28} {cn:>5} {ca:>8.3f} {cb:>8.3f} "
                  f"{d:>+8.3f} {pv:>10.2g}{star}")
        if pooled_ids:
            pa = sum(a[s]["correct"] for s in pooled_ids) / len(pooled_ids)
            pb = sum(b[s]["correct"] for s in pooled_ids) / len(pooled_ids)
            print(f"    {f'[groups with n<{min_n}]':<28} {len(pooled_ids):>5} "
                  f"{pa:>8.3f} {pb:>8.3f} {pb-pa:>+8.3f}")
        print("    * p < 0.05 (McNemar, uncorrected for multiple comparisons)")

    # regressions are the most informative rows in the whole report
    regressions = [s for s in shared if a[s]["correct"] and not b[s]["correct"]]
    if regressions:
        print(f"\n  SAMPLE REGRESSIONS (first 5 of {len(regressions)})")
        for s in regressions[:5]:
            print(f"    [{a[s].get('category','?')}] gold={a[s]['gold_text'][:30]!r} "
                  f"-> now {b[s]['pred_text']!r}")

    return {
        "n": n, "acc_a": acc_a, "acc_b": acc_b, "delta": acc_b - acc_a,
        "gained": gained, "lost": lost, "p_value": p_value,
        "delta_ci": [ci_lo, ci_hi],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze AgroBench predictions")
    ap.add_argument("run", type=Path)
    ap.add_argument("--compare", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    base = load_run(args.run)
    if args.compare:
        mod = load_run(args.compare)
        summary = report_compare(base, mod, args.run.stem, args.compare.stem)
    else:
        summary = report_single(base, top_k=args.top_k)

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[out] {args.json_out}")


if __name__ == "__main__":
    main()
