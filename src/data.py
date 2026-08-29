"""AgroBench data handling: prompt construction and gold-answer resolution.

Deliberately free of torch/transformers imports so the logic here can be unit
tested on a machine with no GPU and no model weights.

AgroBench schema (risashinoda/AgroBench on the Hugging Face Hub):
    image     : PIL image
    question  : str
    options   : list[str]
    answer    : str      -- see resolve_gold_index for why this needs care
    category  : str      -- one of the 7 tasks
    crop       : str
    source    : str
    id        : str
The dataset ships a single split named "train". It is an EVALUATION benchmark:
never train on it.
"""

from __future__ import annotations

import re
import random
import string
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

LETTERS = string.ascii_uppercase

# Our wording: more emphatic about suppressing explanation, which reduces
# unparseable outputs in generate mode.
STRICT_ANSWER_INSTRUCTION = (
    "Answer with the letter of the correct option only. "
    "Do not explain your reasoning."
)

# Verbatim from the AgroBench authors' common/prompts.py. Use this for any
# number you intend to compare against the published results table -- prompt
# wording moves multiple-choice accuracy by a point or two, so a baseline
# built on different wording is not comparable to theirs.
OFFICIAL_ANSWER_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly."
)

ANSWER_INSTRUCTION = STRICT_ANSWER_INSTRUCTION  # backwards compatibility


# --------------------------------------------------------------------------
# normalisation helpers
# --------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    """Aggressive normalisation used only for matching, never for display.

    Handles the ways a gold answer string can fail to byte-match its option:
    unicode dashes, curly quotes, case, punctuation, whitespace runs, and a
    leading enumeration prefix such as "B. " or "(b)".
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)  # dash variants -> hyphen
    s = s.strip()
    s = re.sub(r"^\(?([A-Za-z])[\).:]\s+", "", s)  # strip "B. " / "(b) " prefix
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_bare_letter(s: str, n_options: int) -> int | None:
    """Return an index if `s` looks like a bare option letter, else None."""
    t = str(s).strip().strip("().:").strip()
    if len(t) == 1 and t.upper() in LETTERS:
        idx = LETTERS.index(t.upper())
        return idx if idx < n_options else None
    return None


class GoldResolutionError(ValueError):
    """Raised when a gold answer cannot be tied to exactly one option."""


def resolve_gold_index(answer: str, options: Sequence[str]) -> int:
    """Map the dataset's `answer` field onto an index into `options`.

    The benchmark may store the gold as the full option text or as a bare
    letter, and either form may drift from the option string by punctuation or
    unicode. Resolving this wrong silently corrupts every number downstream,
    so this function is strict: it raises rather than guessing when the match
    is ambiguous.

    Resolution order:
      1. exact string match against an option
      2. bare letter ("B", "(b)", "b.")
      3. normalised match against an option (must be unique)
    """
    if not options:
        raise GoldResolutionError("empty options list")

    opts = list(options)

    # 1. exact
    exact = [i for i, o in enumerate(opts) if str(o) == str(answer)]
    if len(exact) == 1:
        return exact[0]

    # 2. bare letter -- only when the answer is not itself an option string
    letter_idx = _is_bare_letter(answer, len(opts))
    if letter_idx is not None and not exact:
        return letter_idx

    # 3. normalised
    na = normalize_text(answer)
    norm = [i for i, o in enumerate(opts) if normalize_text(o) == na and na != ""]
    if len(norm) == 1:
        return norm[0]

    if len(exact) > 1 or len(norm) > 1:
        raise GoldResolutionError(
            f"answer {answer!r} matches multiple options {opts!r}"
        )
    raise GoldResolutionError(f"answer {answer!r} matches no option in {opts!r}")


# --------------------------------------------------------------------------
# option permutation (for position-bias control)
# --------------------------------------------------------------------------

def permute_options(
    options: Sequence[str], gold_index: int, seed: int, sample_id: str
) -> tuple[list[str], int]:
    """Deterministically shuffle options and remap the gold index.

    Multiple-choice VLM scores are contaminated by position bias: a model that
    leans toward "A" scores above chance without perceiving anything. Re-running
    the eval with a different --shuffle-seed and comparing accuracy tells you
    how much of your number is real. The permutation is keyed on the sample id
    so it is reproducible per sample regardless of iteration order.
    """
    rng = random.Random(f"{seed}:{sample_id}")
    order = list(range(len(options)))
    rng.shuffle(order)
    new_options = [options[i] for i in order]
    new_gold = order.index(gold_index)
    return new_options, new_gold


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

def build_mcq_prompt(question: str, options: Sequence[str],
                     style: str = "official") -> str:
    """Render the question block. The image is attached separately.

    style="official" reproduces the benchmark authors' mcq_prompt exactly:
    no blank lines, their instruction wording. This is the default so the
    headline numbers stay comparable to the published table.

    style="strict" uses our wording with blank-line separation. Worth running
    as a secondary condition -- if the two differ a lot, prompt sensitivity is
    itself a finding.
    """
    if style not in ("official", "strict"):
        raise ValueError(f"unknown prompt style {style!r}")

    lines = [str(question).strip()]
    if style == "strict":
        lines.append("")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {str(opt).strip()}")
    if style == "strict":
        lines.append("")
        lines.append(STRICT_ANSWER_INSTRUCTION)
    else:
        lines.append(OFFICIAL_ANSWER_INSTRUCTION)
    return "\n".join(lines)


def parse_letter_official(text: str, options: Sequence[str]) -> int | None:
    """Reimplementation of the authors' letter_from_text, returning an index.

    Deliberately faithful, including its quirks:
      - the leading-letter regex matches with NO trailing punctuation, so a
        response beginning "Anthracnose" is read as 'A'
      - it falls back to substring-matching an option's text in the response
      - it allows A-E regardless of how many options exist

    Use this when reporting numbers alongside the paper. Use
    parse_letter_from_text for a stricter reading. Reporting both, when they
    differ, tells you how much of the score is parser leniency.
    """
    if not text:
        return None
    upper = str(text).strip().upper()

    m = re.match(r"^\s*([A-E])[\.\:\-\s]*", upper)
    if m:
        idx = LETTERS.index(m.group(1))
        return idx if idx < len(options) else None

    if upper in list(LETTERS[:len(options)]):
        return LETTERS.index(upper)

    for i, opt in enumerate(options):
        if str(opt).lower() in str(text).lower():
            return i
    return None


def parse_letter_from_text(text: str, n_options: int) -> int | None:
    """Best-effort extraction of a choice from free-form generated text.

    Only used in --decode generate mode. Returns None when nothing parses,
    which the evaluator records as an abstention rather than silently
    scoring it wrong -- an unparseable output is a different failure from a
    confident wrong answer, and conflating them hides real problems.
    """
    if not text:
        return None
    t = text.strip()

    # "A", "A.", "(A)", "Answer: A" at the start
    m = re.match(r"^\s*\(?\s*([A-Za-z])\s*[\).:,]?\s*(?:$|\s)", t)
    if m:
        idx = LETTERS.index(m.group(1).upper())
        if idx < n_options:
            return idx

    m = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Za-z])\b", t, re.I)
    if m:
        idx = LETTERS.index(m.group(1).upper())
        if idx < n_options:
            return idx

    # a single standalone capital letter anywhere
    cands = {c for c in re.findall(r"\b([A-Z])\b", t) if LETTERS.index(c) < n_options}
    if len(cands) == 1:
        return LETTERS.index(cands.pop())
    return None


# --------------------------------------------------------------------------
# record container
# --------------------------------------------------------------------------

@dataclass
class Sample:
    sample_id: str
    question: str
    options: list[str]
    gold_index: int
    category: str
    crop: str
    source: str = ""
    image: Any = None
    prompt_style: str = "official"
    meta: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return build_mcq_prompt(self.question, self.options, style=self.prompt_style)

    @property
    def gold_text(self) -> str:
        return self.options[self.gold_index]


def make_sample(
    row: dict,
    index: int,
    shuffle_seed: int | None = None,
    keep_image: bool = True,
    prompt_style: str = "official",
) -> Sample:
    """Build a Sample from a raw dataset row, resolving the gold answer."""
    options = [str(o) for o in row["options"]]
    gold = resolve_gold_index(row["answer"], options)

    # AgroBench reuses the same `id` across several questions about one image,
    # so the raw field is not a key. The row index is appended to make it
    # unique. Iteration order over a fixed dataset is deterministic, so the
    # composite id is stable across runs and machines -- which is what resume
    # and paired comparison both depend on.
    raw_id = row.get("id")
    sid = f"{raw_id}#{index}" if raw_id else f"idx{index}"

    if shuffle_seed is not None:
        options, gold = permute_options(options, gold, shuffle_seed, sid)

    return Sample(
        sample_id=sid,
        question=str(row["question"]),
        options=options,
        gold_index=gold,
        category=str(row.get("category", "unknown")),
        crop=str(row.get("crop", "unknown")),
        source=str(row.get("source", "")),
        image=row.get("image") if keep_image else None,
        prompt_style=prompt_style,
    )


def iter_samples(
    dataset: Iterable[dict],
    categories: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    prompt_style: str = "official",
) -> Iterable[Sample]:
    """Yield Samples, skipping and reporting rows whose gold cannot be resolved."""
    wanted = {c.lower() for c in categories} if categories else None
    wanted_src = {s.lower() for s in sources} if sources else None
    kept = 0
    for i, row in enumerate(dataset):
        if wanted and str(row.get("category", "")).lower() not in wanted:
            continue
        if wanted_src and str(row.get("source", "")).lower() not in wanted_src:
            continue
        try:
            sample = make_sample(
                row, i, shuffle_seed=shuffle_seed, prompt_style=prompt_style
            )
        except GoldResolutionError as exc:
            print(f"[warn] skipping row {i}: {exc}")
            continue
        yield sample
        kept += 1
        if limit is not None and kept >= limit:
            return
