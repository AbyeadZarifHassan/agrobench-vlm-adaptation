r"""Verify the model path end to end without any dataset access.

    python src/smoke_model.py
    python src/smoke_model.py --image path\to\your\photo.jpg

Loads Qwen2-VL-2B in 4-bit, builds a fake multiple-choice question over a
single image, and runs both decoding modes. This exercises everything that can
break -- transformers v5 input building, 4-bit quantization on your GPU, the
letter-token table, memory headroom -- while AgroBench access is still pending.

A synthetic image of coloured shapes is generated if you do not pass one. The
model's answer does not matter much; what matters is that it runs, stays in
memory budget, and returns a valid letter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image, ImageDraw  # noqa: E402


def make_test_image(size: int = 896) -> Image.Image:
    """A plain synthetic image. Deliberately large-ish so the max_pixels cap
    actually has something to clamp."""
    img = Image.new("RGB", (size, size), (232, 240, 226))
    d = ImageDraw.Draw(img)
    d.ellipse([size * 0.15, size * 0.15, size * 0.55, size * 0.55],
              fill=(86, 140, 60))
    d.rectangle([size * 0.5, size * 0.5, size * 0.85, size * 0.85],
                fill=(176, 92, 48))
    for i in range(6):
        x = size * (0.2 + 0.1 * i)
        d.line([(x, size * 0.62), (x, size * 0.92)], fill=(40, 70, 30), width=6)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Model smoke test, no dataset needed")
    ap.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from data import build_mcq_prompt
    from model import load_model, score_by_generation, score_by_letter_logits, set_seed

    print(f"[env] torch {torch.__version__}")
    print(f"[env] transformers {__import__('transformers').__version__}")
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[env] {torch.cuda.get_device_name(0)}  {total:.1f} GB")
    else:
        print("[env] NO CUDA -- this will be extremely slow")

    image = Image.open(args.image).convert("RGB") if args.image else make_test_image()
    print(f"[env] image {image.size}")

    set_seed(0)
    t0 = time.time()
    lm = load_model(
        model_id=args.model_id,
        load_in_4bit=not args.no_4bit,
        max_pixels=args.max_pixels,
    )
    print(f"[ok] model loaded in {time.time() - t0:.1f}s")

    if torch.cuda.is_available():
        print(f"[mem] after load: {torch.cuda.memory_allocated()/1e9:.2f} GB "
              f"allocated, {torch.cuda.max_memory_allocated()/1e9:.2f} GB peak")

    options = ["Early blight", "Late blight", "Powdery mildew", "Bacterial spot"]
    prompt = build_mcq_prompt(
        "Which disease is most likely shown in this image?", options
    )
    print("\n--- prompt ---")
    print(prompt)
    print("--------------\n")

    t0 = time.time()
    logit_out = score_by_letter_logits(lm, image, prompt, len(options))
    dt_logit = time.time() - t0
    print(f"[ok] logit mode  -> {options[logit_out['pred_index']]!r} "
          f"(conf {logit_out['confidence']:.3f})  {dt_logit:.2f}s")
    print(f"     letter scores: {logit_out['letter_scores']}")

    t0 = time.time()
    gen_out = score_by_generation(lm, image, prompt, len(options))
    dt_gen = time.time() - t0
    parsed = (options[gen_out["pred_index"]]
              if gen_out["pred_index"] is not None else "UNPARSEABLE")
    print(f"[ok] generate mode -> {parsed!r}  {dt_gen:.2f}s")
    print(f"     raw output: {gen_out['raw_output']!r}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[mem] peak {peak:.2f} GB of {total:.1f} GB "
              f"({peak/total:.0%})")
        if peak / total > 0.85:
            print("[warn] very little headroom. Lower --max-pixels before the "
                  "full run, since AgroBench images vary in size.")

    n = 1502
    print(f"\n[estimate] at {dt_logit:.2f}s/sample, the {n}-sample disease "
          f"identification set takes about {n * dt_logit / 60:.0f} min in logit mode")

    if logit_out["pred_index"] == gen_out["pred_index"]:
        print("[ok] both decoding modes agree on this sample")
    else:
        print("[note] the two modes disagree here. Fine on one synthetic image; "
              "worth watching if it persists across real samples.")


if __name__ == "__main__":
    main()
