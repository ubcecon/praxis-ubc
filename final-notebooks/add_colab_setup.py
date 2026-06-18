# Add's a Colab setup cell that pulls the notebook's data folder when run on Colab. Integrated into Comet_main.yml
import json
import sys
from pathlib import Path

def _docs_dir(ipynb_path):
    # Repo-relative folder of the notebook on the praxis-notebooks branch, e.g. docs/AMNE-376
    parts = Path(ipynb_path).parts
    i = parts.index("docs")
    return "/".join(parts[i:-1])

def _setup_cell(dir_rel):
    src = [
        "# praxis-colab-setup: on Colab, pull this notebook's data folder so relative paths work\n",
        "try:\n",
        "    import google.colab\n",
        "    !git clone --depth 1 --filter=blob:none --sparse -b praxis-notebooks https://github.com/ubcecon/praxis-ubc.git /content/praxis-ubc\n",
        f"    !git -C /content/praxis-ubc sparse-checkout set {dir_rel}\n",
        f"    %cd /content/praxis-ubc/{dir_rel}\n",
        "except ImportError:\n",
        "    pass\n",
    ]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def add_setup(ipynb_path):
    dir_rel = _docs_dir(ipynb_path)
    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    # Place after the title markdown so the setup cell is the first thing users run
    pos = 1 if cells[0]["cell_type"] == "markdown" else 0
    cells.insert(pos, _setup_cell(dir_rel))

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Added Colab setup cell to {ipynb_path} (cd {dir_rel})")

if __name__ == "__main__":
    add_setup(sys.argv[1])
