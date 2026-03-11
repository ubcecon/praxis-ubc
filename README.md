# Image Analysis Workshop (share-ai)

This branch contains the `image_analysis` workshop materials, including:

- Notebook: `image_analysis/image_analysis_demo.ipynb`
- Data and precomputed assets under `image_analysis/data/`
- Figures and media under `image_analysis/media/`

## Launch on UBC Open JupyterHub (CWL Required)

Use the link below to pull this branch and open the workshop directory directly in JupyterLab:

- [Open on UBC JupyterHub](https://open.jupyter.ubc.ca/jupyter/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2Fubcecon%2Fshare-ai&branch=image_analysis_demo&urlpath=lab%2Ftree%2Fshare-ai%2Fimage_analysis)

Notes:

- This requires a UBC CWL login.
- The link targets branch `image_analysis_demo`.
- It opens the `image_analysis/` directory so users can run the notebook immediately.

## Running the Notebook Normally

Open and run:

- `image_analysis/image_analysis_demo.ipynb`

The notebook includes a setup cell that checks and installs missing packages inside the notebook environment.
If packages are installed during that step, restart the kernel/runtime and run the setup cell once more.

## Colab Setup (For Users Without UBC Access)

You can open the notebook directly in Colab:

- [Open in Colab](https://colab.research.google.com/github/ubcecon/share-ai/blob/image_analysis_demo/image_analysis/image_analysis_demo.ipynb)

After Colab opens:

1. Run all cells from top to bottom.
2. If the notebook setup cell installs packages, allow Colab to restart the runtime.
3. Re-run the setup cell once after restart, then continue with the notebook.

## Resource Expectations

- The repository includes workshop data and precomputed embeddings used by the notebook.
- OCR and deep-learning sections rely on packages listed in `requirements.txt`.
- On fresh environments, the first setup run may take longer while dependencies initialize.
