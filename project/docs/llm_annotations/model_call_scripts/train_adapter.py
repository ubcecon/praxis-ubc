"""Fine-tune one QLoRA adapter on a Civil Comments sample. Needs the GPU.

    . ./scripts/env.ps1
    pyenv model_call_scripts/train_adapter.py --train-csv data/civil_train.csv         --adapter qlora-toxicity --tag plain

About 31 minutes for 200 steps on an RTX 2070 SUPER. Every 20 steps it does two things:
pushes the probe set through the model and saves what it recorded, and writes a checkpoint
to resume from. Pass --resume to pick up a killed run from the last of those.

Recipe: r=16, alpha=32, dropout 0.05, seven target modules, batch 2 with accumulation 8,
lr 2e-4 cosine, fp16, gradient checkpointing, 512-token cap.
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_call_scripts import gpu_run  # noqa: E402
from model_call_scripts.cached_run import (  # noqa: E402
    DATA, MODEL_ID, RUBRIC, SEED, read_set,
)

ARTIFACTS = DATA / "artifacts"
ADAPTERS = gpu_run.ADAPTERS
MAX_SEQ = 512
PROBE_EVERY = 20


class PeakRam(threading.Thread):
    """Watch host RAM: Windows spills VRAM into it instead of raising out of memory."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, False
        self.peak_rss = self.peak_sys = 0

    def run(self):
        proc = psutil.Process()
        while not self.stop_flag:
            self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
            self.peak_sys = max(self.peak_sys, psutil.virtual_memory().used)
            time.sleep(self.interval)

    def snapshot(self):
        return round(self.peak_rss / 2**30, 2), round(self.peak_sys / 2**30, 2)


class LabelledComments(Dataset):
    """One training row per comment, with the prompt masked out of the loss."""

    def __init__(self, texts, labels, tok):
        self.rows = []
        for text, label in zip(texts, labels):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": RUBRIC.format(comment=text)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            answer = f"LABEL: {'yes' if label == 1 else 'no'}{tok.eos_token}"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(answer, add_special_tokens=False)["input_ids"]
            # Truncate the comment, never the answer: cutting the tail off the prompt
            # removes the "reply in this format" instruction and the model rambles.
            if len(p_ids) + len(a_ids) > MAX_SEQ:
                p_ids = p_ids[: MAX_SEQ - len(a_ids)]
            ids = p_ids + a_ids
            self.rows.append({"input_ids": ids,
                              "labels": [-100] * len(p_ids) + a_ids,
                              "attention_mask": [1] * len(ids)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch, pad_id):
    width = max(len(b["input_ids"]) for b in batch)
    out = {k: [] for k in ("input_ids", "labels", "attention_mask")}
    for b in batch:                                    # right-pad for training
        gap = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * gap)
        out["labels"].append(b["labels"] + [-100] * gap)
        out["attention_mask"].append(b["attention_mask"] + [0] * gap)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class ProbeCallback(TrainerCallback):
    """Every PROBE_EVERY steps, record what the model answers and what it is thinking."""

    def __init__(self, model, tok, comments, labels, out):
        self.model, self.tok, self.comments = model, tok, comments
        self.labels, self.out = labels, out
        self.steps, self.p_yes, self.hidden = [], [], []

    def capture(self, step):
        t0 = time.time()
        was_training = self.model.training
        self.model.eval()
        p, h = gpu_run.probe(self.model, self.tok, self.comments, RUBRIC)
        if was_training:
            self.model.train()
        self.steps.append(step)
        self.p_yes.append(p)
        self.hidden.append(h)
        self.flush()
        print(f"[probe] step {step:>3}  mean P(yes) {p.mean():.4f}  "
              f"{time.time() - t0:.0f}s  peak VRAM "
              f"{torch.cuda.max_memory_allocated() // 2**20} MiB", flush=True)

    def flush(self):
        """Rewrite the npz after every probe, so a killed run keeps what it reached."""
        if self.out is None:
            return
        np.savez_compressed(
            self.out,
            steps=np.array(self.steps, dtype=np.int64),
            hidden=np.stack(self.hidden).astype(np.float16),
            p_yes=np.stack(self.p_yes).astype(np.float32),
            labels=self.labels,
        )

    def on_train_begin(self, args, state, control, **kw):
        self.capture(0)

    def on_step_end(self, args, state, control, **kw):
        if state.global_step % PROBE_EVERY == 0:
            self.capture(int(state.global_step))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--adapter", required=True, help="directory name under models/adapters")
    ap.add_argument("--tag", required=True, help="suffix for the artifacts this writes")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--optim", default="paged_adamw_8bit")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint of a killed run")
    args = ap.parse_args()

    watch = PeakRam()
    watch.start()

    train_df = pd.read_csv(ROOT / args.train_csv, encoding="utf-8-sig",
                           dtype={"comment_id": str})
    probe_set = read_set("civil_probe.csv")
    toxic = (train_df.is_toxic == "yes")
    print(f"[data] {len(train_df)} training comments, {toxic.sum()} toxic "
          f"({toxic.mean():.0%}), {train_df.article_id.nunique()} articles")
    print(f"       {len(probe_set)} probe comments, from articles this never sees")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    train_ds = LabelledComments(train_df.comment_text.tolist(), toxic.astype(int).tolist(),
                                tok)

    # transformers honours the checkpoint's own quantization config over anything passed in, and this one asks for bfloat16, which Turing cannot do. Mutate it before loading.
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.quantization_config["bnb_4bit_compute_dtype"] = "float16"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=cfg, dtype=torch.float16, device_map={"": 0})
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    torch.cuda.reset_peak_memory_stats()
    targs = TrainingArguments(
        # Outside the repo. Nothing is saved here (save_strategy="no"), but the Trainer creates the directory regardless and the repo's cache/ holds model answers only.
        output_dir=str(ADAPTERS.parent.parent / "scratch" / "trainer_toxicity"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.steps // 20),   # transformers 5 dropped warmup_ratio
        logging_steps=PROBE_EVERY,
        # Checkpoints exist only to resume from, so two is enough. The probe npz is what carries the history section 5 plots.
        save_strategy="steps",
        save_steps=PROBE_EVERY,
        save_total_limit=2,
        report_to=[],
        fp16=True, bf16=False,                   # Turing: fp16, never bf16
        optim=args.optim,
        gradient_checkpointing=True,
        seed=SEED,
        dataloader_num_workers=0,                # Windows: workers fork poorly
    )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    probe_out = ARTIFACTS / f"probe_checkpoints_{args.tag}.npz"
    probe_cb = ProbeCallback(model, tok, probe_set.comment_text.tolist(),
                             (probe_set.is_toxic == "yes").astype(int).to_numpy(),
                             probe_out)
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id),
                      callbacks=[probe_cb])

    scratch = Path(targs.output_dir)
    resume = None
    if args.resume:
        found = sorted(scratch.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        resume = str(found[-1]) if found else None
        print(f"[resume] {resume or 'nothing to resume from, starting at step 0'}")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_seconds = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() // 2**20
    rss, sys_ram = watch.snapshot()
    watch.stop_flag = True
    watch.join()

    out_dir = ADAPTERS / args.adapter
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    probe_cb.flush()

    log = {
        "seed": SEED, "max_seq_length": MAX_SEQ,
        "train_csv": args.train_csv, "adapter": args.adapter,
        "n_train": len(train_ds), "share_toxic": round(float(toxic.mean()), 3),
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "trainable_params": trainable,
                 "trainable_pct": round(100 * trainable / total, 3)},
        "precision": "fp16 (bf16 disabled, Turing sm_75 has no native bf16)",
        "optimizer": args.optim,
        "per_device_batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "steps": args.steps,
        "history": [{"step": h["step"], "loss": h["loss"]}
                    for h in trainer.state.log_history if "loss" in h],
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(train_seconds / args.steps, 2),
        "final_loss": round(result.training_loss, 4),
        "peak_vram_mib": peak_vram,
        "vram_total_mib": torch.cuda.get_device_properties(0).total_memory // 2**20,
        "peak_rss_gib": rss, "peak_system_ram_gib": sys_ram,
        "probe_steps": probe_cb.steps,
    }
    (ARTIFACTS / f"train_log_{args.tag}.json").write_text(json.dumps(log, indent=2),
                                                          encoding="utf-8")
    print(f"\n[ok] adapter   -> {out_dir}")
    print(f"[ok] artifacts -> train_log_{args.tag}.json, probe_checkpoints_{args.tag}.npz")
    print(f"     {train_seconds / 60:.1f} min, {train_seconds / args.steps:.2f} s/step, "
          f"peak VRAM {peak_vram} MiB of {log['vram_total_mib']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
