# Running the model

The notebook does not need any of this. It reads saved answers out of `cache/` and runs in seconds. This is only relevant if you want to regenerate those answers, train a new adapter, or rebuild the notebook from source or make edits to deeper code content. 

This file is the handoff. It covers what is in the directory, how to reproduce every committed artifact, what you can safely change, and what will quietly break if you change it.

## How this directory is split

Two halves, with different audiences.

- The **student half** is `llm_toxicity_annotation.ipynb` plus `data/`, `cache/` and `media/`. It runs on any laptop with no GPU and no network, because every model answer it needs is already saved. This is the forward facing part. 
- The **internal half** is `model_call_scripts/` plus this file plus the adapter on HuggingFace. It regenerates the first half. Nothing here runs without a GPU except `build_data.py` and `paths.py`.

A reader of the notebook never touches the second half. This is the goal: if the notebook ever needs a GPU to render, the teaching artifact is broken, it will never run on Collab, JupyterOpen etc. 

## What is in this directory

```
llm_toxicity_annotation.ipynb    the notebook, 43 cells
Running_on_GPU.md                this file
cache/                           model answers, keyed by model + adapter + prompt hash
data/                            the sampled corpora the notebook reads
data/artifacts/                  training log and probe checkpoints
media/                           four plots, written by the notebook itself
model_call_scripts/              everything that needs a GPU or the network
```

## What you need

A CUDA GPU with 8 GB of memory. Scoring peaks at roughly 2 GB and training at 4.4 GB, so 8 GB leaves room and 6 GB would very likely work.

Python 3.11 or newer, plus the packages below. These are the versions that produced the cache currently in the repo.

```
torch==2.13.0+cu126          # match the cu wheel to your driver
transformers==5.15.0
peft==0.20.0
bitsandbytes==0.50.0
accelerate==1.14.0
huggingface_hub==1.27.0
pandas==3.0.5
numpy==2.4.6
psutil==7.2.2
```

Install torch from the PyTorch index that matches your CUDA version, then install the rest normally.

The `unsloth` package is not required. The model id contains `unsloth` because Unsloth published the 4-bit build these scripts load, but no code here imports their library. Loading happens through `transformers` and `peft` alone.

## Where the big files go

Three downloads are kept outside the repo: the base model at 1.4 GB, the adapter at 78 MB, and the Wikipedia Detox source files at 114 MB. 

Keeping them out is deliberate, and matches the repository guideline to avoid committing large files. Both corpora are handled the same way: download the source, sample it, commit the sample.

To relocate them, copy `model_call_scripts/paths.env.example` to `model_call_scripts/paths.env` and uncomment the entries you want to change. `paths.env` is gitignored, so your local paths stay local. To see what the scripts resolved, and whether those locations exist yet, run:

```
python -s model_call_scripts/paths.py
```

That command also reports your console encoding and warns you if user site-packages is enabled, both of which are covered under Known traps below.

## The two models

The base model is fetched automatically the first time a script needs it, so it requires no action:

```
unsloth/Qwen3-1.7B-unsloth-bnb-4bit
```

The adapter is a separate 78 MB download, published at `Alexr951/qlora-toxicity-qwen3-1.7b`. It belongs in the directory that `paths.py` reports as `adapters`, inside a folder called `qlora-toxicity`:

```
hf download Alexr951/qlora-toxicity-qwen3-1.7b --local-dir ADAPTERS/qlora-toxicity
```

The equivalent from Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("Alexr951/qlora-toxicity-qwen3-1.7b",
                  local_dir="ADAPTERS/qlora-toxicity")
```

Replace `ADAPTERS` with the path `paths.py` printed. That folder name matters: `gpu_run.py` looks the adapter up by the name stored in `cached_run.ADAPTER`, so renaming the directory makes it invisible to the scripts.

## Running it

Every step is resumable. Work that has already been written is skipped on the next run, so an interrupted job is not completely damaging, it just picks up more-or-less where it left off.

**1. Build the data.** Downloads both corpora and writes the csv files into `data/`.

```
python -s model_call_scripts/build_data.py
```

**2. Measure the speed before committing to a full pass.** Times 50 comments and extrapolates from there. Writes nothing to disk.

```
python -s model_call_scripts/gpu_run.py --calibrate 50
```

On an RTX 2070 SUPER the base model manages about 120 comments a minute against roughly 550 for the fine-tuned one, because training taught it to stop once it has given the answer rather than continuing into a paragraph of explanation.

**3. Score both models** across the evaluation, probe and transfer sets.

```
python -s model_call_scripts/gpu_run.py
```

At this point the notebook has everything it reads. Steps 4 and 5 apply only if you are training a replacement adapter.

**4. Train.** 200 steps. Produces the adapter itself, a training log, and the probe checkpoints that section 5 of the notebook plots.

```
python -s model_call_scripts/train_adapter.py --train-csv data/civil_train.csv \
    --adapter qlora-toxicity --tag plain
```

A checkpoint is written every 20 steps, so pass `--resume` to restart a run that died partway.

**5. Score the new adapter** by repeating step 3.

## Changing things

**A different prompt.** Edit `RUBRIC` in `cached_run.py`, then rerun step 3. The prompt is hashed into the cache filename, so the existing answers become unreachable the moment you save the file and the notebook will raise on import until you have rescored. Budget about 22 minutes.

**A different sample size or class balance.** The knobs are all at the top of `cached_run.py`: `N_EVAL_PER_CLASS`, `N_PROBE_PER_CLASS`, `N_TRAIN`, `TOXIC_BAR` and `SEED`. Changing any of them means rerunning step 1 and then step 3, because the comment ids in the csv files will no longer match the ids in the cache.

**A different base model.** `MODEL_ID` appears in both `cached_run.py` and `gpu_run.py` and the two must agree, because it is part of the cache filename in one and the thing being loaded in the other. Expect to retrain: an adapter is tied to the model it was trained against.

**A new adapter alongside the current one.** Give `train_adapter.py` a new `--adapter` name and a new `--tag`, which keeps the artifacts separate. To make the notebook read it, change `ADAPTER` in `cached_run.py`. Adapter name is part of the cache filename, so the old answers stay valid and the new ones land beside them.

**Retraining the same adapter.** `train_adapter.py` overwrites the adapter directory and the tagged artifacts, but `gpu_run.py` will not notice: the cache key contains the adapter name, not its contents, so stale answers from the previous adapter of the same name are still considered valid. Delete the matching `cache/*qlora-toxicity*.json` before rescoring.

## Do not break these

- `RUBRIC` is hashed into the cache filename. This is deliberate, so answers written under one prompt are never read back as though they matched another.
- `SEED` is `20260810` and is used for the article split and every sample. The committed csv files depend on it.
- The split is by **article**, not by comment, so no article contributes to both training and evaluation. Splitting by comment would let the model see other comments from the same thread and inflate the score. `build_data.py` asserts this and will stop if it ever stops holding.
- The csv files are written and read as `utf-8-sig`. Excel reads a csv without the byte-order mark as cp1252 and turns every curly quote into mojibake, so both ends have to agree.
- `comment_id` is read as a string throughout, so ids never lose leading zeros.
- The adapter directory name must match `cached_run.ADAPTER`.

## Known traps

Below is a list of stuff I ran into which might help (or might not):

bf16 does not work on Turing cards even though `torch.cuda.is_bf16_supported()` returns True, because it is counting emulation. Everything here runs in fp16, and `train_adapter.py` overwrites the bfloat16 request in the Unsloth checkpoint's config before loading it.

An oversized batch on Windows spills VRAM into host RAM rather than raising an error, which makes the run roughly thirty times slower instead of stopping it. When a step looks wrong, compare the rate against the figures in step 2 rather than waiting for an exception that will not arrive.

Editing `RUBRIC` makes the existing cache unreachable, by design. The prompt text is hashed into the cache filename, so altering a single character orphans every saved answer and the notebook raises an error on import. Rescoring after a prompt change costs about 22 minutes.

User site-packages can shadow the environment you installed into. On this machine an old `huggingface_hub` in the user directory takes priority over the one in the project environment, and `transformers` then fails on `ImportError: cannot import name 'is_offline_mode'`. Running Python with `-s` disables user site-packages for that process, which is why every command above includes the flag. Python resolves site-packages before any user code runs, so no script can correct this from the inside.

`paths` has to be imported ahead of `transformers`. `HF_HOME` is read once, when `huggingface_hub` is first imported, so assigning it later has no effect and produces no warning, and the 1.4 GB base model lands in your default user cache rather than the location you configured. The scripts import `paths` first for this reason, and call `paths.check_import_order()`, which prints a warning if that ordering is ever disturbed.

Windows consoles default to cp1252, which cannot encode about a dozen of the comments in the sampled data. `paths.py` switches stdout and stderr to UTF-8 on import to keep a stray print from ending a long run.

The training-time probe uses the training sequence cap of 512 rather than the 1024 used for generation, and applies no separate comment trim. A handful of probe comments assemble to more than 512 tokens, so changing this moves them, and with them the kappa that section 5 plots.

## Licences

Civil Comments and Wikipedia Detox are both released CC0, text included, which is why the sampled comments can ship inside `data/`. Qwen3-1.7B is Apache 2.0, and the adapter is a derivative of Qwen3 and carries the same licence.
