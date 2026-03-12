# Image Analysis Workshop

This branch contains the `image_analysis` workshop materials, including:

- Notebook: `image_analysis/image_analysis_demo.ipynb`
- Data and precomputed assets under `image_analysis/data/`
- Figures and media under `image_analysis/media/`
- Workshop slides `image_analysis/image_analysis_slides.slides.html`

## Launch on UBC Open JupyterHub (CWL Required)

Use the link below to pull this branch and open the workshop directory directly in JupyterLab:

- [Open on UBC JupyterHub](https://open.jupyter.ubc.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fubcecon%2Fshare-ai&branch=image_analysis_demo&urlpath=lab%2Ftree%2Fshare-ai%2Fimage_analysis)
- Select the **Tensorflow Notebook** for a faster setup.

Notes:

- This requires a UBC CWL login.
- The link targets branch `image_analysis_demo`.
- It opens the `image_analysis/` directory so you can run the notebook immediately.

## Running the Notebook Normally

Open and run:

- `image_analysis/image_analysis_demo.ipynb`

The notebook includes a setup cell that checks and installs missing packages inside the notebook environment.
If packages are installed during that step, restart the kernel/runtime and run the setup cell once more.

## Colab Setup (For Users Without UBC Access)

You can open the notebook directly in Colab:

- [Open in Colab](https://colab.research.google.com/github/ubcecon/share-ai/blob/image_analysis_demo/image_analysis/image_analysis_demo.ipynb)

**Important**: opening from the link alone does not automatically copy the full repository folders into the Colab runtime.

After Colab opens, add and run this as a new first code cell:

```python
import os
import shutil
from pathlib import Path

repo_dir = Path("/content/share-ai")
if repo_dir.exists():
	shutil.rmtree(repo_dir)

!git clone --depth 1 --branch image_analysis_demo https://github.com/ubcecon/share-ai.git /content/share-ai
os.chdir("/content/share-ai/image_analysis")

print("Working directory:", os.getcwd())
print("Data folder exists:", Path("data").exists())
print("Media folder exists:", Path("media").exists())
```

Then continue with the notebook in this order:

1. Run the notebook setup/install cell.
2. If Colab asks to restart the runtime, allow it.
3. Re-run the bootstrap cell above.
4. Re-run the notebook setup cell, then run the remaining cells.

If you prefer, after running the bootstrap cell you can open the local copy at:

- `File` -> `Open notebook` -> `/content/share-ai/image_analysis/image_analysis_demo.ipynb`

## Static Rendered Notebook

A static [rendered notebook](https://github.com/ubcecon/praxis-ubc/blob/image_analysis_demo/image_analysis_demo.html) in `.html` format is also available in this repository, you can download and view the demo in your web browser locally. 

You may still be able to play with the interactive visualizations using this version, but you cannot **modify and run codes** yourself.

## Workshop Slides

For those interested, you can also download and view the [workshop slides](https://github.com/ubcecon/praxis-ubc/blob/image_analysis_demo/image_analysis/image_analysis_slides.slides.html) in your web browser locally.

## Resource Expectations

- The repository includes workshop data and precomputed embeddings used by the notebook.
- OCR and deep-learning sections rely on packages listed in `requirements.txt`.
- On fresh environments, the first setup run may take longer while dependencies initialize.
