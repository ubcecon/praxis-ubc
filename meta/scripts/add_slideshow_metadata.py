# Sets slideshow.slide_type on cells so the target notebooks open as RISE slideshows.
# Runs in comet_main.yml after add_colab_setup.py, once per notebook.
# Usage: python add_slideshow_metadata.py <notebook.ipynb>
import json
import sys
from pathlib import Path

# Notebooks to make slideshows, by repo-relative path.
TARGETS = {
    "docs/SOCI-280/soci_280_bert.ipynb",
    "docs/hist_workshop/text_embeddings_workshop.ipynb",
}

def _slide_type(cell):
    text = "".join(cell["source"])
    if text.startswith("# praxis-colab-setup"):  # the Colab setup cell
        return "skip"
    if cell["cell_type"] == "markdown" and text.lstrip().startswith("#"):  # a heading
        return "slide"
    return None

def add_slides(ipynb_path):
    if not any(Path(ipynb_path).as_posix().endswith(t) for t in TARGETS):
        return

    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        slide_type = _slide_type(cell)
        if slide_type:
            cell.setdefault("metadata", {})["slideshow"] = {"slide_type": slide_type}

    nb["metadata"]["rise"] = {"autolaunch": False}

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Added slideshow metadata to {ipynb_path}")

if __name__ == "__main__":
    add_slides(sys.argv[1])
