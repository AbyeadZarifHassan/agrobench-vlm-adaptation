# Modifying a Vision-Language Model for Agricultural Disease Identification

Parameter efficient adaptation (QLoRA) of **Qwen2-VL-2B-Instruct** for plant
disease identification, evaluated against the unmodified model on three
datasets under an identical protocol, with paired significance testing.

All experiments ran on a single **RTX 4060 (8 GB)**.

## Results

| Evaluation set | n | Baseline | Modified | Δ | McNemar p |
|---|---|---|---|---|---|
| PlantVillage (lab, in-domain) | 590 | 0.307 | **0.931** | +0.624 | 5.6e-97 |
| PlantDoc (field, same 14 classes) | 482 | 0.411 | **0.737** | +0.326 | 1.9e-34 |
| AgroBench - disease identification | 1,502 | 0.342 | **0.424** | +0.083 | 3.3e-09 |
| AgroBench - all 7 tasks | 4,335 | 0.469 | **0.502** | +0.033 | 6.4e-07 |

The adapter was trained only on lab-photographed images of 15 classes across
three crops. It transfers to field photography of the same classes (0.737) and
to a benchmark spanning 682 diseases and 203 crops that it never saw (0.424 on
disease identification), while leaving AgroBench's six non-disease tasks
statistically unchanged.

**Full write-up, including controls and limitations: [RESULTS.md](RESULTS.md).**

## Is the improvement real?

Three checks, because a large accuracy delta on multiple choice is easy to
manufacture by accident:

**Image-derangement control.** Re-running the best model with every question
paired with a *different* sample's image drops accuracy from 0.931 to 0.349.
The result is visually grounded. The residual sits ~0.10 above chance, so the
visually-grounded gain is ~+0.58, not the full +0.62 which reported that way in
RESULTS.md.

**Paired significance testing.** Every comparison uses exact McNemar on the
intersection of sample ids, with bootstrap CIs on the difference and an explicit
count of regressions. Two accuracy numbers side by side are not evidence.

**Protocol alignment.** The prompt is byte-identical to the AgroBench authors'
`mcq_prompt`, asserted by a unit test. Decoding, seed and vision token budget
are held fixed across every compared run and recorded per run in a
`.config.json`.

## Method

- **Base:** Qwen2-VL-2B-Instruct, 4-bit NF4, bfloat16 compute
- **Adaptation:** LoRA r=16, α=32, dropout 0.05; lr 1e-4 cosine, 3% warmup;
  batch 1 × grad-accum 8; 533 steps; AdamW-8bit; gradient checkpointing
- **Training data:** PlantVillage train split only (4,267 MCQ items, 15 classes)
- **Ablation:** language-side, vision-side, and both
- **Scoring:** logits over option letters at the first answer position - one
  forward pass, deterministic, immune to output-format drift

The ablation is worth noting: language-side (0.886) and vision-side (0.895)
adaptation performed about equally, which did **not** match the prediction from
the baseline error analysis. See RESULTS.md §4.1.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python tests/test_logic.py        # 83 offline tests, no GPU needed
```

Build evaluation sets from a classification dataset:

```bash
python src/build_mcq.py --path PlantVillageDataset/train_val_test/test \
    --out data/pv_test.jsonl --max-per-class 40
```

Baseline, train, evaluate, compare:

```bash
python src/evaluate.py  --run-name baseline --local-jsonl data/pv_test.jsonl
python src/analyze.py   outputs/baseline.jsonl
python src/train_lora.py --train data/pv_train.jsonl --out checkpoints/lora_both --target both
python src/evaluate.py  --run-name lora_both --local-jsonl data/pv_test.jsonl \
    --adapter-path checkpoints/lora_both
python src/analyze.py   outputs/baseline.jsonl --compare outputs/lora_both.jsonl
```

AgroBench (gated - request access on the Hub first, then `hf auth login`):

```bash
python src/evaluate.py --run-name agro_baseline
python src/evaluate.py --run-name agro_lora --adapter-path checkpoints/lora_both
python src/analyze.py  outputs/agro_baseline.jsonl --compare outputs/agro_lora.jsonl
```

## Layout

```
src/data.py             prompt construction, gold-answer resolution, option permutation
src/model.py            model loading, letter-logit scoring, greedy generation
src/evaluate.py         evaluation loop, resume, run-config provenance, controls
src/analyze.py          metrics, error analysis, McNemar comparison
src/train_lora.py       QLoRA training with selectable adaptation target
src/build_mcq.py        classification dataset -> MCQ, with cross-dataset label mapping
src/export_official.py  export predictions in the AgroBench authors' format
src/smoke_model.py      single-image sanity check, no dataset required
tests/test_logic.py     83 offline tests
```

Datasets and checkpoints are not committed. PlantVillage and PlantDoc are on
Kaggle; AgroBench is gated on the Hugging Face Hub.

## Data notes

Strict gold-answer resolution (raising rather than guessing on an unmatched
answer) surfaced seven malformed AgroBench items - answer strings with no
matching option, and one item where the same option appears twice. These are
excluded rather than silently scored wrong. Some AgroBench options are also
nomenclatural synonyms, so a portion of measured error is naming convention
rather than misidentification. Details in RESULTS.md §5.

## Acknowledgements

AgroBench: Shinoda, Inoue, Kataoka, Onishi and Ushiku, *AgroBench:
Vision-Language Model Benchmark in Agriculture*, ICCV 2025, pp. 7634–7644. The
evaluation protocol here follows their released code.

## License

MIT
