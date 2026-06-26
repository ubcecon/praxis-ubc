# Sets slideshow.slide_type on cells so the target notebooks open as RISE slideshows.
# Quarto packs several headings into one markdown cell and RISE only breaks slides at cell
# boundaries, so this splits markdown cells too: each heading starts its own cell and slide.
# Runs in comet_main.yml after add_colab_setup.py, once per notebook.
# Usage: python add_slideshow_metadata.py <notebook.ipynb>
import json
import re
import sys
from pathlib import Path

# Notebooks to make slideshows, by repo-relative path.
TARGETS = {
    "docs/SOCI-280/soci_280_bert.ipynb",
    "docs/hist_workshop/text_embeddings_workshop.ipynb",
}

HEADING = re.compile(r"^#{1,6}\s", re.M)  # an ATX heading at the start of a line

def _split_markdown(cell):
    # One piece per heading. Text before the first heading stays on the previous slide.
    lines = "".join(cell["source"]).splitlines(keepends=True)
    pieces, cur = [], []
    for line in lines:
        if cur and HEADING.match(line):
            pieces.append(cur)
            cur = []
        cur.append(line)
    pieces.append(cur)

    cells = []
    for piece in pieces:
        c = {"cell_type": "markdown", "metadata": {}, "source": piece}
        if HEADING.match(piece[0]):
            c["metadata"]["slideshow"] = {"slide_type": "slide"}
        cells.append(c)
    return cells

def add_slides(ipynb_path):
    if not any(Path(ipynb_path).as_posix().endswith(t) for t in TARGETS):
        return

    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = []
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and text.startswith("# praxis-colab-setup"):
            cell["metadata"]["slideshow"] = {"slide_type": "skip"}  # hide the Colab setup cell
            cells.append(cell)
        elif cell["cell_type"] == "markdown" and HEADING.search(text):
            cells.extend(_split_markdown(cell))
        else:
            cells.append(cell)
    nb["cells"] = cells
    nb["metadata"]["rise"] = {"autolaunch": False}

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Added slideshow metadata to {ipynb_path}")

if __name__ == "__main__":
    add_slides(sys.argv[1])
