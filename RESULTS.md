# Modifying a Vision-Language Model for Agricultural Disease Identification

**Abyead Zarif Hassan** — evaluation task for Dr. Md Sultan Mahmud, University of Georgia
August 2026

---

## Summary

I adapted Qwen2-VL-2B-Instruct using parameter-efficient fine-tuning (QLoRA) on
plant disease images and compared it against the unmodified model on three
evaluation sets under an identical protocol. The modification produced
statistically significant improvements on all three, with the largest gains in
the trained domain and no significant degradation on unrelated agricultural
tasks.

| Evaluation set | Images | Baseline | Modified | Δ | McNemar p |
|---|---|---|---|---|---|
| PlantVillage (lab, in-domain) | 590 | 0.307 | **0.931** | +0.624 | 5.6e-97 |
| PlantDoc (field, same classes) | 482 | 0.411 | **0.737** | +0.326 | 1.9e-34 |
| AgroBench — disease id | 1,502 | 0.342 | **0.424** | +0.083 | 3.3e-09 |
| AgroBench — all 7 tasks | 4,335 | 0.469 | **0.502** | +0.033 | 6.4e-07 |

All work ran on a single RTX 4060 (8 GB).

---

## 1. Setup

**Base model.** Qwen2-VL-2B-Instruct, loaded in 4-bit NF4 with double
quantisation, bfloat16 compute. A 2B model was chosen because it trains and
evaluates within 8 GB, allowing a full ablation rather than a single run on a
larger model that would not fit.

**Training data.** PlantVillage (3-crop subset: tomato, potato, bell pepper;
15 classes). The dataset's own train/val/test split was used, and only the
training split fed adaptation. 4,267 multiple-choice items.

**Evaluation protocol.** Every question is presented as an image plus a
lettered multiple-choice list. The prompt is byte-identical to the AgroBench
authors' `mcq_prompt`, and there is a unit test asserting that so it cannot
drift. Answers are read by comparing logits over the option letters at the
first answer position — one forward pass, fully deterministic, immune to
output-format variation. Greedy decoding, fixed seed, and an identical vision
token budget across every run being compared.

**Distractor construction.** For the datasets I built from classification
labels, distractors are drawn preferentially from the *same crop* as the gold
answer, so a tomato question offers other tomato diseases rather than a corn
disease. Roughly 70% of items have all-same-crop options. This keeps the task
about fine-grained disease discrimination rather than crop recognition; the
alternative would have made the benchmark trivially easy.

---

## 2. Diagnosing the baseline before modifying anything

I ran the unmodified model first and analysed its errors, so that the choice of
modification would follow from evidence rather than guesswork.

Baseline accuracy on PlantVillage was 0.307 (95% CI [0.271, 0.345]) against
0.250 chance. Three things stood out:

**Confidence carried no information.** Accuracy was flat across confidence
buckets — 0.242 at 0.4–0.5 confidence, 0.385 at 0.85+. A model using visual
evidence should be right more often when confident.

**One answer dominated.** "Tomato late blight" appeared as the prediction in
six of the twelve most frequent confusion pairs. Even *healthy potato* was
called late blight 17 times. The output distribution was anchored on the most
familiar disease name rather than on the image.

**Letter bias.** 32% of predictions were "B" against a 24.7% gold rate.

The picture was a model failing to use the image at all, rather than one seeing
the symptom but naming it wrongly. That suggested the language-side prior was
the bottleneck — a hypothesis the ablation went on to complicate (§4).

---

## 3. The modification

LoRA adapters (rank 16, α 32, dropout 0.05) trained on the multiple-choice task
itself: image and lettered options in, answer letter out. Loss is computed on
the answer token only; the prompt is masked, so the gradient signal is about
choosing correctly rather than reproducing the question.

Three variants, differing only in which modules are adapted:

- **language** — the LLM's attention and MLP projections (196 modules, 18.5 M
  trainable, 1.49% of the model)
- **vision** — the vision tower's blocks
- **both**

Hyperparameters: lr 1e-4 with cosine decay and 3% warmup, batch size 1 with
gradient accumulation 8, 533 optimiser steps, AdamW-8bit, gradient
checkpointing. Peak memory 2.69 GB; 79 minutes per run.

---

## 4. Results

### 4.1 In-domain (PlantVillage test split, 590 questions)

| Model | Accuracy | 95% CI | Δ vs baseline | p |
|---|---|---|---|---|
| Baseline | 0.307 | [0.271, 0.345] | — | — |
| + LoRA (language) | 0.886 | — | +0.580 | 4.6e-83 |
| + LoRA (vision) | 0.895 | — | +0.588 | 9.9e-82 |
| + LoRA (both) | **0.931** | [0.907, 0.948] | +0.624 | 5.6e-97 |

**The ablation did not confirm my hypothesis.** I predicted from the error
analysis that language-side adaptation would carry the improvement, since the
baseline's failure looked like a language prior overriding the image. Vision-
side adaptation alone performed just as well (0.895 vs 0.886). Either pathway
recovers most of the gain, which suggests the baseline's failure was not
specifically a language-prior problem but that neither pathway had been adapted
to this discrimination at all.

Combining both adds a small but significant increment over language alone
(+0.044, p = 3.1e-04), indicating the two adaptations capture partly
overlapping signal.

### 4.2 Control: is the model actually using the image?

A model can score well on multiple choice by learning which answers tend to be
correct. To test this, I re-ran the best model with every question paired with a
*different* sample's image — a seeded derangement, with same-class donors
avoided where possible.

| Condition | Accuracy |
|---|---|
| Correct images | 0.931 |
| Deranged images | **0.349** |

Accuracy falls by 58 points, confirming the result is visually grounded.

The residual 0.349 sits 0.099 above chance, so roughly ten points of the
headline number come from text-side structure — the option set partially
reveals the crop, and class priors do some work. **The visually-grounded gain
is therefore about +0.58, not the full +0.62.**

### 4.3 Out-of-domain transfer (PlantDoc, 482 questions)

PlantVillage images are lab-photographed: one leaf, uniform background,
controlled lighting. PlantDoc contains real field photographs of the same 14
classes — cluttered backgrounds, variable lighting, multiple leaves per frame.
The option vocabulary was held identical to the training set so that any drop
measures domain shift rather than vocabulary mismatch.

| Model | Accuracy | Δ | p |
|---|---|---|---|
| Baseline | 0.411 | — | — |
| + LoRA (both) | **0.737** | +0.326 | 1.9e-34 |

173 gains against 16 regressions. Accuracy falls from 0.931 in-domain to 0.737
on field imagery, so about 19 points of the in-domain performance does not
survive the change in imaging conditions — but the adapted model still sits far
above the 0.411 baseline. Given the well-documented tendency of
PlantVillage-trained models to fail on field images, this degrades more
gracefully than expected.

### 4.4 AgroBench (4,335 questions across 7 tasks)

AgroBench spans 203 crops and 682 diseases in field conditions, plus tasks on
weeds, pests, management and machinery. The adapter saw none of this — it was
trained on 15 lab classes from three crops.

| Task | n | Baseline | Modified | Δ | p |
|---|---|---|---|---|---|
| **did** — disease identification | 1,502 | 0.342 | 0.424 | **+0.083** | 3.3e-09 |
| dmn — disease management | 568 | 0.676 | 0.695 | +0.019 | 0.23 |
| wid — weed identification | 609 | 0.279 | 0.291 | +0.011 | 0.55 |
| pid — pest identification | 544 | 0.504 | 0.515 | +0.011 | 0.62 |
| cmn — crop management | 409 | 0.531 | 0.533 | +0.002 | 1.00 |
| mqa | 301 | 0.698 | 0.694 | −0.003 | 1.00 |
| tm — tools & machinery | 402 | 0.654 | 0.647 | −0.007 | 0.76 |
| **Overall** | 4,335 | 0.469 | 0.502 | +0.033 | 6.4e-07 |

**The gain is confined to the trained domain.** Disease identification improves
significantly; the other six tasks show no significant change in either
direction. Narrow adaptation improved its target without measurable collateral
damage to unrelated capabilities — a cleaner outcome than a large
across-the-board shift, which would have suggested the adapter was exploiting
answer-format regularities rather than learning disease features.

By pathogen category, the gains concentrate where the baseline was weakest:

| Category | n | Baseline | Modified | Δ | p |
|---|---|---|---|---|---|
| Phytoplasma | 117 | 0.248 | 0.513 | +0.265 | 3.7e-08 |
| Virus | 149 | 0.161 | 0.396 | +0.235 | 7.9e-08 |
| Bacteria | 628 | 0.344 | 0.446 | +0.102 | 1.0e-06 |
| Fungus | 1,137 | 0.540 | 0.547 | +0.007 | 0.66 |

The baseline was already strongest on fungal disease and gained nothing there;
viral and phytoplasma diseases, near or below chance before, improved most.

**Weed identification is the hardest task** (0.279 baseline, 0.291 after) and
barely moved — consistent with the benchmark authors' own finding that
open-source VLMs struggle most on weed identification.

---

## 5. Notes on benchmark data quality

Gold-answer resolution in my harness raises an error rather than guessing when
an answer string cannot be matched to exactly one option. This surfaced seven
malformed AgroBench items, including:

- an answer of "Removing surface weeds" against an option reading "Removing
  surface weed"
- "Stone picker" with no matching option in the list
- one item where the same option text appears twice and the gold matches both

These rows were excluded rather than silently scored as wrong. Separately, some
options are nomenclatural synonyms — one regression was gold `Redshank
(Persicaria maculosa)` scored against the model's `Polygonum persicaria
(Redshank)`, the same plant under a different binomial. A portion of measured
error on AgroBench is therefore naming convention rather than misidentification.

---

## 6. Limitations

**Lab-to-field gap remains.** 0.931 on lab images against 0.737 on field images
of the same classes. Deployment performance would be closer to the latter.

**Narrow training vocabulary.** 15 classes from three crops. The AgroBench
result shows transfer beyond them, but the ceiling is modest (0.424 on 682
disease classes).

**Ten points of the in-domain number are not visual.** Established by the
derangement control and reported rather than absorbed into the headline figure.

**One seed per configuration.** Time constraints meant no repeat runs, so
run-to-run variance is unmeasured. The effect sizes are far larger than
plausible seed variance, but this should be stated.

**Single base model.** Whether these findings hold for larger VLMs is untested;
8 GB of VRAM sets the ceiling.

**Multiple comparisons.** Per-task and per-category p-values are uncorrected.
The primary comparisons (overall accuracy per dataset) are pre-specified; the
breakdowns are exploratory.

---

## 7. Reproducibility

All results are produced by scripts in the accompanying repository.

- 83 offline unit tests covering gold-answer resolution, prompt construction,
  option permutation, answer parsing, and the statistical functions. No GPU
  required: `python tests/test_logic.py`
- Every evaluation writes a `.config.json` recording model, adapter, decoding
  mode, vision token budget, seed and dataset, so two runs can be checked for
  comparability rather than assumed comparable.
- Predictions stream to JSONL and runs resume after interruption; scoring is
  separate from inference, so results can be re-scored without re-running the
  model.
- `export_official.py` converts predictions to the AgroBench authors'
  `predictions.jsonl` format so their scorer can be run on the same outputs.

**One measurement bug found and fixed during the work.** AgroBench reuses the
same `id` across several questions about one image. Because paired comparison
keys on sample id, duplicates silently overwrote each other and the comparison
was computed on an arbitrary one-per-id subset. This inflated the apparent
AgroBench effect from +0.033 to +0.158. Sample ids now include the row index,
and the analysis script raises an error if a prediction file contains
duplicates. The corrected figure is the one reported above.

---

## 8. What I would do next

*[Zarif — replace this section with your own priorities. Some candidates:]*

- Train on field imagery (PlantDoc, or AgroBench-adjacent data) rather than lab
  images, and measure whether the lab-to-field gap closes.
- Test whether raising the vision token budget helps the fine-grained
  confusions that remain — after adaptation the errors are species-level
  (Beet Armyworm → Fall Armyworm), which is a different failure from the
  baseline's language prior.
- Repeat with multiple seeds to bound run-to-run variance.
- Extend to weed identification, the weakest task, where the headroom is
  largest.
