"""Model loading and answer extraction for AgroBench evaluation.

Compatible with transformers v4.45+ and v5.x. The input-building path differs
between them, so _build_inputs tries the v5 API first and falls back once,
caching whichever worked.

Two decoding modes:

  logit  (default) -- one forward pass; read the logits at the position where
          the answer letter would be generated and take the argmax over the
          option letters only. Fully deterministic, ~2x faster than generation,
          and immune to output-format drift. Because the model is forced to
          choose among valid letters it cannot abstain, so this measures
          discrimination rather than instruction-following.

  generate -- greedy generation plus a parser. Closer to the published
          AgroBench protocol and to how the model would really be used, but
          introduces parse failures, reported separately as abstentions.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from data import LETTERS, parse_letter_from_text, parse_letter_official

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class LoadedModel:
    model: torch.nn.Module
    processor: object
    letter_token_ids: dict[int, list[int]]
    model_id: str
    quantized: bool
    # "v5" | "v4" | None -- resolved on the first sample, then reused
    input_path: str | None = field(default=None)


def _pick_attn_impl() -> str:
    """flash_attention_2 saves real memory on an 8GB card, but only if the
    package is present and the GPU is Ampere or newer. sdpa is the safe
    fallback and still far better than eager."""
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"
    if not torch.cuda.is_available():
        return "sdpa"
    major, _ = torch.cuda.get_device_capability()
    return "flash_attention_2" if major >= 8 else "sdpa"


def load_model(
    model_id: str = DEFAULT_MODEL,
    load_in_4bit: bool = True,
    max_pixels: int | None = 512 * 28 * 28,
    min_pixels: int | None = 64 * 28 * 28,
    adapter_path: str | None = None,
    device_map: str = "auto",
    verbose: bool = True,
) -> LoadedModel:
    """Load a VLM for evaluation.

    max_pixels is the single most important memory knob for Qwen2-VL. Its
    dynamic resolution will happily turn one high-res field photo into
    thousands of vision tokens and OOM an 8GB card. 512*28*28 caps it at
    roughly 512 vision tokens. RAISE THIS DELIBERATELY as an experimental
    variable if you are testing the fine-grained-perception hypothesis -- but
    then it must be raised for the baseline too, or the comparison is invalid.
    """
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # transformers < 4.45
        from transformers import AutoModelForVision2Seq as AutoVLM

    # Qwen-family processors take the pixel budget as kwargs; others reject it.
    proc_kwargs = {}
    if max_pixels is not None:
        proc_kwargs["max_pixels"] = max_pixels
    if min_pixels is not None:
        proc_kwargs["min_pixels"] = min_pixels
    try:
        processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)
    except (TypeError, ValueError) as exc:
        if verbose and proc_kwargs:
            print(f"[warn] processor rejected the pixel budget "
                  f"({exc.__class__.__name__}); loading without it. "
                  "Memory use will be higher.")
        processor = AutoProcessor.from_pretrained(model_id)

    if verbose:
        _report_pixel_budget(processor, max_pixels)

    quant_config = None
    if load_in_4bit:
        # transformers v5 removed the from_pretrained(load_in_4bit=True)
        # shortcut. Passing a BitsAndBytesConfig works on both v4 and v5.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # v5 defaults dtype to "auto" (whatever the checkpoint was saved in), so
    # pass it explicitly to keep runs reproducible across versions.
    load_kwargs = dict(
        quantization_config=quant_config,
        device_map=device_map,
        attn_implementation=_pick_attn_impl(),
    )
    try:
        model = AutoVLM.from_pretrained(model_id, dtype=torch.bfloat16, **load_kwargs)
    except TypeError:
        # v4.x before the torch_dtype -> dtype rename
        model = AutoVLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, **load_kwargs
        )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)
    return LoadedModel(
        model=model,
        processor=processor,
        letter_token_ids=_build_letter_token_ids(tokenizer),
        model_id=model_id,
        quantized=load_in_4bit,
    )


def _report_pixel_budget(processor, requested: int | None) -> None:
    """Confirm the cap actually took effect. If the processor silently ignored
    it, the first high-resolution image will OOM and the cause will not be
    obvious from the traceback."""
    ip = getattr(processor, "image_processor", None)
    actual = getattr(ip, "max_pixels", None)
    if actual is None and ip is not None:
        size = getattr(ip, "size", None)
        if isinstance(size, dict):
            actual = size.get("longest_edge") or size.get("max_pixels")
    if requested and actual and actual != requested:
        print(f"[warn] requested max_pixels={requested} but the processor "
              f"reports {actual}. Watch memory on the first few samples.")
    elif actual:
        print(f"[info] max_pixels = {actual}")


def _build_letter_token_ids(tokenizer, max_options: int = 26) -> dict[int, list[int]]:
    """Collect the token ids that can begin each answer letter.

    A letter can tokenise differently with and without a leading space, and
    some tokenizers split it further. We keep every distinct FIRST token id and
    score a letter by the best of its variants, so the comparison across
    letters stays fair.
    """
    table: dict[int, list[int]] = {}
    for i in range(max_options):
        letter = LETTERS[i]
        ids = set()
        for variant in (letter, f" {letter}", f"{letter}.", f"({letter}"):
            try:
                enc = tokenizer.encode(variant, add_special_tokens=False)
            except Exception:
                continue
            if enc:
                ids.add(int(enc[0]))
        if ids:
            table[i] = sorted(ids)
    return table


# --------------------------------------------------------------------------
# input building -- the one place v4 and v5 genuinely differ
# --------------------------------------------------------------------------

def _messages(image, prompt: str, inline_image: bool) -> list[dict]:
    """v5 wants the image object inside the message content; v4 wants a bare
    placeholder in the template with the image passed separately."""
    img_part = {"type": "image", "image": image} if inline_image else {"type": "image"}
    return [{"role": "user", "content": [img_part, {"type": "text", "text": prompt}]}]


def _build_inputs_v5(lm: LoadedModel, image, prompt: str):
    """transformers v5: apply_chat_template tokenizes and returns a BatchEncoding."""
    return lm.processor.apply_chat_template(
        _messages(image, prompt, inline_image=True),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _build_inputs_v4(lm: LoadedModel, image, prompt: str):
    """transformers v4: render the template to text, then pass images separately."""
    text = lm.processor.apply_chat_template(
        _messages(image, prompt, inline_image=False),
        tokenize=False,
        add_generation_prompt=True,
    )
    return lm.processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    )


def _build_inputs(lm: LoadedModel, image, prompt: str):
    """Try the v5 path, fall back to v4, and remember which one worked.

    Resolved once per process rather than per sample: a try/except on every one
    of 1,500 samples would be slow and would bury a real error in noise.
    """
    image = image.convert("RGB")

    if lm.input_path == "v5":
        inputs = _build_inputs_v5(lm, image, prompt)
    elif lm.input_path == "v4":
        inputs = _build_inputs_v4(lm, image, prompt)
    else:
        try:
            inputs = _build_inputs_v5(lm, image, prompt)
            lm.input_path = "v5"
            print("[info] using the transformers v5 chat-template input path")
        except Exception as exc_v5:
            try:
                inputs = _build_inputs_v4(lm, image, prompt)
                lm.input_path = "v4"
                print("[info] using the transformers v4 chat-template input path")
            except Exception as exc_v4:
                raise RuntimeError(
                    "Could not build model inputs with either API.\n"
                    f"  v5 path failed: {exc_v5!r}\n"
                    f"  v4 path failed: {exc_v4!r}"
                ) from exc_v4

    device = lm.model.device
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }


@torch.inference_mode()
def score_by_letter_logits(lm: LoadedModel, image, prompt: str, n_options: int) -> dict:
    """One forward pass; argmax over option letters at the first answer position."""
    inputs = _build_inputs(lm, image, prompt)
    logits = lm.model(**inputs).logits[0, -1, :].float()

    scores: dict[int, float] = {}
    for i in range(n_options):
        ids = lm.letter_token_ids.get(i, [])
        valid = [t for t in ids if t < logits.shape[0]]
        scores[i] = max((logits[t].item() for t in valid), default=float("-inf"))

    pred = max(scores, key=scores.get)
    ordered = torch.tensor([scores[i] for i in range(n_options)])
    probs = torch.softmax(ordered, dim=-1)
    return {
        "pred_index": int(pred),
        "confidence": float(probs[pred]),
        "letter_scores": {LETTERS[i]: round(scores[i], 4) for i in range(n_options)},
        "raw_output": None,
    }


@torch.inference_mode()
def score_by_generation(
    lm: LoadedModel,
    image,
    prompt: str,
    n_options: int,
    options: list[str] | None = None,
    parser: str = "official",
    max_new_tokens: int = 12,
) -> dict:
    """Greedy generation plus a parser. Sampling knobs are omitted rather than
    passed as None -- v5 validates the generation config more strictly, and any
    sampling would make the baseline/modified comparison unreproducible."""
    inputs = _build_inputs(lm, image, prompt)
    out = lm.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    trimmed = out[0][prompt_len:]
    tokenizer = getattr(lm.processor, "tokenizer", lm.processor)
    text = tokenizer.decode(trimmed, skip_special_tokens=True).strip()

    if parser == "official":
        opts = options if options is not None else [""] * n_options
        pred = parse_letter_official(text, opts)
    else:
        pred = parse_letter_from_text(text, n_options)

    return {
        "pred_index": pred,  # None means unparseable -> abstention
        "confidence": None,
        "letter_scores": None,
        "raw_output": text,
    }
