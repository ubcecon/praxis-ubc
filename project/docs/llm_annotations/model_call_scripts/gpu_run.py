"""Run the models on the GPU and save every answer into cache/.

    python -s model_call_scripts/gpu_run.py --calibrate 50   # time it before committing
    python -s model_call_scripts/gpu_run.py                  # score all three sets

Saves the label the model writes for each comment, keyed by model, adapter and prompt text,
so the notebook replays them with no GPU. Resumable: anything already in cache/ is skipped.

Three facts about this machine that cost time to rediscover:

  torch.cuda.is_bf16_supported() returns True on a Turing card but is counting emulation,
  so everything here is fp16. transformers ignores a BitsAndBytesConfig when the checkpoint
  carries its own, so the config object gets mutated before the load. And Windows spills
  VRAM into host RAM instead of raising, so an oversized batch runs thirty times slower
  rather than crashing. Watch the rate, not for an exception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# paths must precede transformers: HF_HOME is read the moment huggingface_hub is imported, and setting it any later is ignored in silence.
from model_call_scripts import paths  # noqa: E402

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from model_call_scripts.cached_run import ADAPTER, RUBRIC, read_set  # noqa: E402

paths.check_import_order()

CACHE_DIR = ROOT / "cache"
ADAPTERS = paths.ADAPTERS

MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
MAX_COMMENT_TOKENS = 400
MAX_SEQ_LENGTH = 1024      # generation
TRAIN_SEQ_LENGTH = 512     # training, and therefore the probe
MAX_NEW_TOKENS = 96
BATCH_SIZE = 8

_LOADED: dict[tuple[str, str | None], tuple] = {}


# ---- loading
def load(adapter_name: str | None = None):
    """Base model, optionally with a LoRA adapter from `models/adapters/<name>`."""
    key = (MODEL_ID, adapter_name)
    if key in _LOADED:
        return _LOADED[key]

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # The checkpoint ships bnb_4bit_compute_dtype=bfloat16, and transformers 5 ignores a BitsAndBytesConfig when the checkpoint carries its own. Turing has no native bf16, so the config object is mutated directly.
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    if getattr(cfg, "quantization_config", None):
        cfg.quantization_config["bnb_4bit_compute_dtype"] = "float16"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, dtype=torch.float16, device_map={"": 0},
    )
    if adapter_name:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTERS / adapter_name))
    model.eval()
    model.generation_config.max_length = None
    _LOADED[key] = (tok, model)
    return tok, model


def unload() -> None:
    """Free the GPU. Two resident copies plus training activations exceed 8 GB, and
    Windows answers that by spilling to host RAM instead of raising."""
    import gc

    _LOADED.clear()
    gc.collect()
    torch.cuda.empty_cache()


# ---- building the prompt
def trim(tok, comment: str) -> str:
    ids = tok(str(comment), add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_COMMENT_TOKENS:
        return str(comment)
    return tok.decode(ids[:MAX_COMMENT_TOKENS])


def raw_prompt(tok, prompt: str, comment: str) -> str:
    """The chat-formatted prompt with the comment untouched."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt.format(comment=str(comment))}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def build_prompt(tok, prompt: str, comment: str) -> str:
    """As above, but with the comment trimmed first -- the generation path."""
    return raw_prompt(tok, prompt, trim(tok, comment))


# ---- cache keys
def cache_path(prompt: str, adapter_name: str | None = None) -> Path:
    """Same key the notebook's reader builds: model, adapter name, prompt hash."""
    slug = MODEL_ID.replace("/", "__") + (f"__{adapter_name}" if adapter_name else "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{slug}__{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.json"


def read_cache(prompt: str, adapter_name: str | None = None) -> dict:
    p = cache_path(prompt, adapter_name)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ---- reading the model's reply
_LABEL_RE = re.compile(r"label\s*[:\-]?\s*\**\s*(yes|no)\b", re.I)


def parse(text: str) -> tuple[str | None, str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = _LABEL_RE.search(text)
    label = m.group(1).lower() if m else None
    if label is None:
        m2 = re.search(r"\b(yes|no)\b", text, re.I)
        label = m2.group(1).lower() if m2 else None
    reason = ""
    r = re.search(r"reason\s*[:\-]\s*\**\s*(.+)", text, re.I | re.S)
    if r:
        reason = " ".join(r.group(1).split())[:300]
    elif text:
        reason = " ".join(text.split())[:300]
    return label, reason


@dataclass
class Stats:
    n_total: int = 0
    n_cached: int = 0
    n_generated: int = 0
    seconds: float = 0.0
    peak_vram_mib: int = 0
    unparsed: list[str] = field(default_factory=list)

    @property
    def per_minute(self) -> float:
        return 60 * self.n_generated / self.seconds if self.seconds else float("nan")

    def __str__(self) -> str:
        return (f"{self.n_total} comments | {self.n_cached} cached, "
                f"{self.n_generated} generated in {self.seconds:.1f}s "
                f"({self.per_minute:.1f}/min) | peak VRAM {self.peak_vram_mib} MiB "
                f"| unparsed {len(self.unparsed)}")


# ---- labelling
def annotate(comments, comment_ids, prompt, adapter_name=None, write=True):
    """Label every comment, generating only the ones not already saved."""
    comments = [str(c) for c in comments]
    comment_ids = [str(i) for i in comment_ids]
    path = cache_path(prompt, adapter_name)
    cache = read_cache(prompt, adapter_name)
    st = Stats(n_total=len(comments))

    todo = [i for i, cid in enumerate(comment_ids) if cid not in cache]
    st.n_cached = len(comments) - len(todo)

    if todo:
        tok, model = load(adapter_name)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for s in range(0, len(todo), BATCH_SIZE):
            idx = todo[s: s + BATCH_SIZE]
            texts = [build_prompt(tok, prompt, comments[i]) for i in idx]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_SEQ_LENGTH).to("cuda")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                raw = tok.decode(gen[j], skip_special_tokens=True)
                label, reason = parse(raw)
                if label is None:
                    st.unparsed.append(raw[:200])
                cache[comment_ids[i]] = {"label": label, "reason": reason, "raw": raw}
        st.seconds = time.time() - t0
        st.n_generated = len(todo)
        st.peak_vram_mib = torch.cuda.max_memory_allocated() // 2**20
        if write:
            path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")

    return [cache[cid]["label"] for cid in comment_ids], st


# ---- the training-time probe
LABEL_PREFIX = "LABEL:"


def yes_no_ids(tok) -> tuple[int, int]:
    """The two tokens that can follow "LABEL:" — taken from the real strings so the
    leading space is whatever the tokenizer actually produces."""
    base = tok(LABEL_PREFIX, add_special_tokens=False)["input_ids"]
    out = []
    for word in (" yes", " no"):
        ids = tok(LABEL_PREFIX + word, add_special_tokens=False)["input_ids"]
        assert ids[:len(base)] == base, "label prefix did not tokenise as a prefix"
        out.append(ids[len(base)])
    return out[0], out[1]


@torch.inference_mode()
def probe(model, tok, comments, prompt, batch_size=2, max_length=None):
    """P("yes") and the final-layer hidden state at the answer position.

    Note the sequence cap: the probe runs inside the training loop, so it uses the
    *training* limit of 512 with no separate comment trim, not the 1024 used for
    generation. A handful of probe comments assemble to more than 512 tokens, and getting
    this wrong moves them, and with them the recorded kappa.

    Only the last position is ever materialised: hidden states come from a hook on the
    input to the output embedding, and `logits_to_keep=1` keeps the logit tensor to one
    row. Asking for `output_hidden_states=True` instead materialises all 29 layers and
    risks a VRAM spill, which on Windows shows up as a 30x slowdown rather than an error.
    """
    max_length = TRAIN_SEQ_LENGTH if max_length is None else max_length
    yes_id, no_id = yes_no_ids(tok)
    captured: list[torch.Tensor] = []

    def hook(_module, args, _output=None):
        captured.append(args[0][:, -1, :].detach().float().cpu())

    handle = model.get_output_embeddings().register_forward_pre_hook(hook)
    side = tok.padding_side
    tok.padding_side = "left"
    p_yes, hidden = [], []
    try:
        for s in range(0, len(comments), batch_size):
            chunk = comments[s: s + batch_size]
            texts = [raw_prompt(tok, prompt, c) + LABEL_PREFIX for c in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to("cuda")
            captured.clear()
            out = model(**enc, logits_to_keep=1)
            logits = out.logits[:, -1, :].float()
            pair = torch.stack([logits[:, yes_id], logits[:, no_id]], dim=-1)
            p_yes.append(torch.softmax(pair, dim=-1)[:, 0].cpu())
            hidden.append(captured[-1])
    finally:
        handle.remove()
        tok.padding_side = side

    return (torch.cat(p_yes).numpy().astype("float32"),
            torch.cat(hidden).numpy().astype("float16"))


# ---- P(yes) scoring
SCORE_SEQ_LENGTH = 1024   # measured max assembled single-comment prompt is 498


def pyes_cache_path(prompt: str, adapter_name: str | None = None) -> Path:
    """Same keying as the label cache, in its own namespace."""
    slug = MODEL_ID.replace("/", "__") + (f"__{adapter_name}" if adapter_name else "")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{slug}__pyes__{hashlib.sha256(prompt.encode()).hexdigest()[:16]}.json"


def score_pyes(comments, comment_ids, prompt, adapter_name=None, batch_size=8,
               max_length=SCORE_SEQ_LENGTH, write=True):
    """P("yes") per comment, generating only what is not already saved.

    One forward pass per comment rather than a generation, which is the only reason
    a set this size fits the budget.
    """
    import numpy as np

    comments = [str(c) for c in comments]
    comment_ids = [str(i) for i in comment_ids]
    path = pyes_cache_path(prompt, adapter_name)
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    todo = [i for i, cid in enumerate(comment_ids) if cid not in cache]
    st = Stats(n_total=len(comments), n_cached=len(comments) - len(todo))

    if todo:
        tok, model = load(adapter_name)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for s in range(0, len(todo), batch_size):
            idx = todo[s: s + batch_size]
            p, _ = probe(model, tok, [comments[i] for i in idx], prompt,
                         batch_size=len(idx), max_length=max_length)
            for j, i in enumerate(idx):
                cache[comment_ids[i]] = float(p[j])
        st.seconds = time.time() - t0
        st.n_generated = len(todo)
        st.peak_vram_mib = torch.cuda.max_memory_allocated() // 2**20
        if write:
            path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    return np.array([cache[c] for c in comment_ids], dtype="float32"), st


# ---- the driver
VARIANTS = {"base model": None, "fine-tuned": ADAPTER}


def sets_to_score():
    """Every (name, comments, ids) the GPU has to cover."""
    out = []
    for label, name in (("evaluation set", "civil_eval.csv"),
                        ("probe set", "civil_probe.csv"),
                        ("wikipedia detox", "detox_eval.csv")):
        frame = read_set(name)
        out.append((label, frame.comment_text, frame.comment_id))
    return out


def calibrate(n):
    """Measure the generation rate before committing to the whole pass.

    The estimate that decides whether this is a coffee break or an afternoon depends
    entirely on tokens per second on this card, and that is not worth guessing. Nothing is
    written to the cache: this runs the base model over n comments and reports the rate.
    """
    evaluation = read_set("civil_eval.csv")
    comments = list(evaluation.comment_text)[:n]
    ids = list(evaluation.comment_id)[:n]
    print(f"calibrating on {len(comments)} comments, nothing will be written\n")

    t0 = time.time()
    _, st = annotate(comments, ids, RUBRIC, None, write=False)
    gen_rate = st.per_minute
    print(f"  generation  {st}")

    _, stp = score_pyes(comments, ids, RUBRIC, None, write=False)
    pyes_rate = stp.per_minute
    print(f"  P(yes)      {stp}")

    total = sum(len(c) for _, c, _ in sets_to_score())
    n_variants = len(VARIANTS)
    gen_min = total * n_variants / gen_rate if gen_rate else float("nan")
    pyes_min = total * n_variants / pyes_rate if pyes_rate else float("nan")
    print(f"\n  {total:,} comments x {n_variants} variants = {total * n_variants:,} of each")
    print(f"  projected: {gen_min:.0f} min generating + {pyes_min:.0f} min scoring "
          f"= {(gen_min + pyes_min) / 60:.1f} hours")
    print("  plus training, measured at about 24 minutes per adapter")
    print(f"  (measured in {time.time() - t0:.0f}s of wall clock)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="time N comments on the base model and project, writing nothing")
    ap.add_argument("--only", choices=list(VARIANTS),
                    help="score one variant instead of both")
    ap.add_argument("--pyes", action="store_true",
                    help="also save P(yes) per comment; nothing in the notebook reads it")
    args = ap.parse_args()

    if args.calibrate:
        return calibrate(args.calibrate)

    work = sets_to_score()
    variants = {args.only: VARIANTS[args.only]} if args.only else VARIANTS
    total = sum(len(c) for _, c, _ in work) * len(variants)
    print(f"{total:,} comment-model pairs to cover, before anything already cached\n")

    started = time.time()
    for name, adapter in variants.items():
        print(f"===== {name} =====")
        for set_name, comments, ids in work:
            _, st = annotate(comments, ids, RUBRIC, adapter)
            print(f"  labels   {set_name:<16} {st}")
            if st.unparsed:
                print(f"        first unparsed: {st.unparsed[0][:120]}")
            if args.pyes:
                _, st = score_pyes(comments, ids, RUBRIC, adapter)
                print(f"  P(yes)   {set_name:<16} {st}")
        unload()       # two resident copies do not fit; Windows spills rather
        print()                # than raising, which looks like a hang

    print(f"done in {(time.time() - started) / 60:.1f} min")
    for path in sorted(CACHE_DIR.glob("*.json")):
        print(f"  {path.name}  {path.stat().st_size / 1024:,.0f} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
