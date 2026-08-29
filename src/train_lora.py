"""QLoRA fine-tuning of a VLM on multiple-choice agricultural disease data.

    python src/train_lora.py --train data/pv_train.jsonl --out checkpoints/lora_lang
    python src/train_lora.py --train data/pv_train.jsonl --out checkpoints/lora_vis \
        --target vision

The model is trained on exactly the task it is evaluated on: image + question +
lettered options in, answer letter out. Only the answer token carries loss --
the prompt is masked -- so the gradient signal is about choosing correctly, not
about reproducing the question.

WHICH MODULES TO ADAPT is the experimental variable:

  --target language  LoRA on the LLM's attention and MLP projections. Tests the
                     hypothesis that the model SEES the symptom but lacks the
                     agricultural knowledge to name it.

  --target vision    LoRA on the vision tower's blocks. Tests the hypothesis
                     that the bottleneck is fine-grained perception -- that
                     lesion texture separating two similar diseases is being
                     discarded before the language model ever sees it.

  --target both      Both at once, for the ablation's fourth cell.

Run the baseline first, read the error analysis, and let it choose the target.
A modification picked from evidence is a result; one picked at random is a
coin flip you then have to defend.

Deliberately a hand-written loop rather than transformers' Trainer: fewer
moving parts across library versions, and every choice is visible.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image

from data import LETTERS, build_mcq_prompt, resolve_gold_index

# Module-name patterns. Matched against named_modules() at runtime and the hit
# count is printed, so a rename in a future model version shows up as "0
# matched" rather than silently training nothing.
TARGET_PATTERNS = {
    "language": [r"(?<!visual\.)(?:^|\.)(?:q|k|v|o)_proj$",
                 r"(?<!visual\.)(?:^|\.)(?:gate|up|down)_proj$"],
    "vision": [r"visual\..*\.(?:qkv|proj)$",
               r"visual\..*\.mlp\.(?:fc1|fc2)$",
               r"visual\..*\.(?:q|k|v|o)_proj$",
               r"visual\..*\.mlp\.(?:gate|up|down)_proj$"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune a VLM on MCQ data")
    p.add_argument("--train", type=Path, required=True, help="MCQ jsonl")
    p.add_argument("--out", type=Path, required=True, help="adapter output dir")
    p.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--target", choices=["language", "vision", "both"],
                   default="language")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-pixels", type=int, default=512 * 28 * 28,
                   help="MUST match the value used for evaluation")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=500,
                   help="checkpoint every N optimizer steps, so a crash at "
                        "hour two does not cost the whole run")
    return p.parse_args()


def load_mcq(path: Path, limit: int | None, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            options = [str(o) for o in row["options"]]
            try:
                gold = resolve_gold_index(row["answer"], options)
            except Exception as exc:
                print(f"[warn] skipping {row.get('id')}: {exc}")
                continue
            rows.append({
                "image_path": row["image_path"],
                "question": row["question"],
                "options": options,
                "gold_index": gold,
            })
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def select_target_modules(model, target: str) -> list[str]:
    """Resolve regex patterns to concrete module names present in this model."""
    patterns = []
    if target in ("language", "both"):
        patterns += TARGET_PATTERNS["language"]
    if target in ("vision", "both"):
        patterns += TARGET_PATTERNS["vision"]

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and "Linear" not in type(module).__name__:
            continue
        if any(re.search(p, name) for p in patterns):
            names.append(name)

    if not names:
        raise SystemExit(
            f"No modules matched target={target!r}. The model's layer names may "
            "differ from the patterns in TARGET_PATTERNS. Print "
            "[n for n,_ in model.named_modules()] and adjust."
        )

    in_vision = sum(1 for n in names if "visual" in n)
    print(f"[lora] {len(names)} modules matched "
          f"({in_vision} in the vision tower, {len(names)-in_vision} in the LLM)")
    return names


def build_example(processor, row: dict, input_path: str):
    """Tokenize one example and mask everything but the answer letter.

    The prompt length is measured by templating the user turn alone with a
    generation prompt, then everything up to that point is set to -100. Slicing
    by a hardcoded offset would break the moment the chat template changes.
    """
    image = Image.open(row["image_path"]).convert("RGB")
    prompt = build_mcq_prompt(row["question"], row["options"])
    answer = LETTERS[row["gold_index"]]

    if input_path == "v5":
        user = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        full = user + [{"role": "assistant",
                        "content": [{"type": "text", "text": answer}]}]
        prompt_enc = processor.apply_chat_template(
            user, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        full_enc = processor.apply_chat_template(
            full, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt")
    else:
        user = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        full = user + [{"role": "assistant",
                        "content": [{"type": "text", "text": answer}]}]
        p_text = processor.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True)
        f_text = processor.apply_chat_template(
            full, tokenize=False, add_generation_prompt=False)
        prompt_enc = processor(text=[p_text], images=[image], return_tensors="pt")
        full_enc = processor(text=[f_text], images=[image], return_tensors="pt")

    prompt_len = prompt_enc["input_ids"].shape[1]
    labels = full_enc["input_ids"].clone()
    labels[:, :prompt_len] = -100

    batch = dict(full_enc)
    batch["labels"] = labels
    return batch


def main() -> None:
    args = parse_args()

    from model import _build_inputs_v5, load_model, set_seed  # noqa: F401
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)
    rows = load_mcq(args.train, args.max_samples, args.seed)
    if not rows:
        raise SystemExit(f"no usable rows in {args.train}")
    print(f"[data] {len(rows)} training examples from {args.train}")

    lm = load_model(
        model_id=args.model_id,
        load_in_4bit=not args.no_4bit,
        max_pixels=args.max_pixels,
    )
    model, processor = lm.model, lm.processor

    # Detect which chat-template API this transformers version wants, using the
    # first row, before the training loop commits to one.
    input_path = "v5"
    try:
        build_example(processor, rows[0], "v5")
    except Exception as exc_v5:
        try:
            build_example(processor, rows[0], "v4")
            input_path = "v4"
        except Exception as exc_v4:
            raise SystemExit(f"cannot tokenize examples.\n v5: {exc_v5!r}\n"
                             f" v4: {exc_v4!r}")
    print(f"[info] using the transformers {input_path} input path")

    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    target_modules = select_target_modules(model, args.target)
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable/1e6:.2f}M of {total/1e6:.0f}M "
          f"({100*trainable/total:.3f}%)")

    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        print("[optim] AdamW8bit")
    except Exception:
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )
        print("[optim] AdamW (fp32 states)")

    n_steps = max(1, int(len(rows) * args.epochs) // args.grad_accum)
    warmup = max(1, int(n_steps * args.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * step / warmup
        prog = (step - warmup) / max(1, n_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    print(f"[train] {n_steps} optimizer steps, batch 1 x {args.grad_accum} accum, "
          f"warmup {warmup}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_config.json").write_text(json.dumps({
        "model_id": args.model_id, "target": args.target, "rank": args.rank,
        "alpha": args.alpha, "lr": args.lr, "epochs": args.epochs,
        "grad_accum": args.grad_accum, "max_pixels": args.max_pixels,
        "seed": args.seed, "quantized_4bit": not args.no_4bit,
        "n_train": len(rows), "train_file": str(args.train),
        "n_target_modules": len(target_modules),
    }, indent=2))

    model.train()
    device = next(model.parameters()).device
    step = micro = 0
    running: list[float] = []
    start = time.time()
    n_micro_total = int(len(rows) * args.epochs)

    while micro < n_micro_total:
        row = rows[micro % len(rows)]
        try:
            batch = build_example(processor, row, input_path)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running.append(loss.item() * args.grad_accum)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[warn] OOM on {row['image_path']}, skipping")
            optim.zero_grad(set_to_none=True)
            micro += 1
            continue
        except Exception as exc:
            print(f"[warn] skipping {row['image_path']}: {exc}")
            micro += 1
            continue

        micro += 1
        if micro % args.grad_accum == 0:
            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                mean = sum(running[-args.log_every * args.grad_accum:]) / \
                    max(1, len(running[-args.log_every * args.grad_accum:]))
                elapsed = time.time() - start
                eta = (n_steps - step) * elapsed / max(1, step) / 60
                mem = (torch.cuda.max_memory_allocated() / 1e9
                       if torch.cuda.is_available() else 0)
                print(f"  step {step}/{n_steps}  loss {mean:.4f}  "
                      f"lr {lr_at(step):.2e}  peak {mem:.2f}GB  eta {eta:.0f}m")

            if args.save_every and step % args.save_every == 0:
                model.save_pretrained(str(args.out))
                print(f"  [ckpt] saved at step {step}")

    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))
    elapsed = (time.time() - start) / 60
    print(f"\n[done] {step} steps in {elapsed:.1f} min")
    print(f"[out]  {args.out}")
    print(f"[next] python src/evaluate.py --run-name lora_{args.target} "
          f"--adapter-path {args.out} --local-jsonl <your eval file> "
          f"--max-pixels {args.max_pixels}")


if __name__ == "__main__":
    main()
