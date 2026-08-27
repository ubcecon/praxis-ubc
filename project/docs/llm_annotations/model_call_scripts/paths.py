"""Where the big files live. Nothing here is needed to run the notebook.

Checks paths.env first, then the environment, then falls back to a .cache folder inside
this one. Copy paths.env.example to paths.env if the defaults do not suit.

    python model_call_scripts/paths.py     # print what resolved, and whether it exists

Import this before transformers, torch or huggingface_hub. HF_HOME is read once, when
huggingface_hub is first imported, so setting it afterwards is silently ignored and the
1.4 GB base model lands in the default user cache instead. check_import_order() below
says so out loud rather than letting that happen quietly.
"""

import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(__file__).resolve().parent / "paths.env"


def _from_file():
    if not ENV_FILE.exists():
        return {}
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    pairs = (line.split("=", 1) for line in lines
             if "=" in line and not line.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip("\"'") for k, v in pairs}


_FILE = _from_file()


def _get(name, default):
    return Path(_FILE.get(name) or os.environ.get(name) or default)


CACHE = _get("LLM_ANNOT_CACHE", ROOT / ".cache")
ADAPTERS = _get("LLM_ANNOT_ADAPTERS", CACHE / "adapters")
HF_HOME = _get("LLM_ANNOT_HF", CACHE / "huggingface")
DETOX = _get("LLM_ANNOT_DETOX", CACHE / "detox")

# transformers reads HF_HOME when it is imported, so set it before that happens.
os.environ.setdefault("HF_HOME", str(HF_HOME))

# A Windows console defaults to cp1252, and a dozen of the sampled comments contain
# characters it cannot encode, so printing one raises UnicodeEncodeError mid-run.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream.encoding.lower() not in ("utf-8", "utf8"):
            _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass        # already wrapped, or redirected to something without an encoding


def check_import_order():
    """Complain if huggingface_hub was imported before this module set HF_HOME."""
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is None:
        return
    landed = Path(hub.HF_HUB_CACHE)
    if HF_HOME not in landed.parents:
        print(f"[paths] warning: huggingface_hub was already imported, so HF_HOME had no "
              f"effect.\n         wanted {HF_HOME}, downloads will go to {landed}.\n"
              f"         import model_call_scripts.paths before transformers.",
              file=sys.stderr)


if __name__ == "__main__":
    print("from paths.env" if _FILE else "from the environment and defaults")
    for name, path in (("cache", CACHE), ("adapters", ADAPTERS),
                       ("hf_home", HF_HOME), ("detox", DETOX)):
        print(f"  {name:<9} {'ok     ' if path.exists() else 'missing'}  {path}")
    print(f"\nconsole   {sys.stdout.encoding}, printing “café — 你好”")
    if site.ENABLE_USER_SITE:
        print("user site-packages is on. If imports resolve to versions you did not "
              "install, rerun with -s.")
