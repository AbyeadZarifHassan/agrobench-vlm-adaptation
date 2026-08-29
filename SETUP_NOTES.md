# AgroBench VLM Modification Study

Baseline evaluation and error analysis for modifying a vision-language model
and comparing it against the unmodified model on
[AgroBench](https://huggingface.co/datasets/risashinoda/AgroBench)
(Shinoda et al., ICCV 2025).

Target hardware: single RTX 4060, 8 GB VRAM.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_logic.py          # 64 offline checks, no GPU needed
```

The dataset is ~2 GB and downloads on first use to `~/.cache/huggingface`.

## Run order

**1. Look at the data before you model it.**

```bash
python src/evaluate.py --run-name probe --list-categories
```

Prints the seven task names and their sample counts. Use the exact strings it
prints for `--categories`.

**2. Smoke test on 20 samples.** Confirms the model loads in 8 GB, the prompt
format works, and predictions are not degenerate.

```bash
python src/evaluate.py --run-name smoke --limit 20
python src/analyze.py outputs/smoke.jsonl
```

If nearly every prediction is the same letter, stop and fix that before
scaling up — you would just be measuring position bias.

**3. Full baseline.**

```bash
python src/evaluate.py --run-name baseline_full
python src/analyze.py outputs/baseline_full.jsonl --json-out outputs/baseline.json
```

Safe to interrupt; re-running resumes from the existing JSONL.

**4. Position-bias control.** Same model, permuted options. If accuracy drops
sharply, part of your baseline was an artifact and you should report both
numbers.

```bash
python src/evaluate.py --run-name baseline_shuf --shuffle-seed 42
python src/analyze.py outputs/baseline_full.jsonl --compare outputs/baseline_shuf.jsonl
```

**5. Read the error analysis and choose a modification.** See below.

**6. Compare, after training.**

```bash
python src/evaluate.py --run-name lora_v1 --adapter-path checkpoints/lora_v1
python src/analyze.py outputs/baseline_full.jsonl --compare outputs/lora_v1.jsonl
```

## Reading the error analysis

The report is built to discriminate between three different failure modes,
because each implies a different modification:

| Signal in the report | Diagnosis | Modification it justifies |
|---|---|---|
| Few confusion pairs absorb most errors; pairs are visually similar classes | Fine-grained perception | Unfreeze the last vision-tower blocks; raise `--max-pixels` |
| Errors scattered across many classes; high confidence on wrong answers | Missing domain knowledge | LoRA on the language side with external agricultural data |
| Head/tail accuracy gap above ~10 points | Long tail | Class-balanced sampling or reweighted loss |
| One letter dominates predictions | Position bias, not a real skill | Fix before anything else |

Run this before choosing what to change. A modification chosen from evidence is
a result; a modification chosen at random is a coin flip you have to defend.

## Protocol rules

These exist because breaking them is the usual way a comparison silently
becomes meaningless.

1. **Never train on AgroBench.** It is evaluation-only and ships a single split
   confusingly named `train`. Adapt on external data (PlantVillage, IP102,
   AgroInstruct) and keep AgroBench strictly zero-shot.
2. **Hold decoding fixed.** Same `--decode`, same `--max-pixels`, same seed
   across every run you compare. `evaluate.py` writes a `.config.json` next to
   each run; diff them before believing any delta.
3. **Compare paired, not aggregate.** `analyze.py --compare` aligns on
   `sample_id`, runs an exact McNemar test, and reports regressions. An
   accuracy delta with no significance test is not evidence.
4. **Report regressions.** The report lists samples the modification broke.
   Include them in your writeup.

## Memory notes (8 GB)

- `--max-pixels` is the dominant knob. Default `512*28*28` ≈ 512 vision tokens.
  Raising it is a legitimate experiment, but raise it for the baseline too.
- 4-bit NF4 is on by default. `--no-4bit` needs roughly 5 GB for a 2B model in
  bf16 and is worth trying if you have headroom, since quantization slightly
  depresses the baseline.
- A 7B VLM will not fit for training. Use Kaggle's free 30 GPU-hours/week
  (2×T4 16 GB) for that, and debug locally on a 20-sample subset first.

## Files

```
src/data.py       prompt building, gold-answer resolution, option permutation
src/model.py      loading, letter-logit scoring, greedy generation
src/evaluate.py   eval loop, resume, run-config provenance
src/analyze.py    metrics, error analysis, McNemar comparison
tests/test_logic.py   offline tests for everything that does not need a GPU
```
