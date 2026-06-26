# Moving the pre-rendered notebooks onto Quarto freeze

## What this is

Right now the 13 heavy notebooks are not rendered during the build. We render them by hand, save a self contained `.html`, commit that `.html`, and the **.dockerfile** copies it into the site after the Quarto render. The problem is that those `.html` files never go through Quarto again, so anything we change in the theme or the code highlighting never reaches them. 

Quarto freeze fixes this. You render the notebook once on your machine where you have the data, and Quarto saves the *outputs* (the executed cells, the plots) into `project/_freeze/`. From then on the build re-renders the page from that saved output. The code does not run again so the build never needs the data, but the page does go through Quarto, so highlighting, themes and the launch button all apply normally. It is also much smaller in the repo. The old `political_economy.html` was 8.3 MB, its freeze cache is 56 KB.

You cannot `.gitignore` the freeze cache. The build needs it, so it has to be committed like the `.html` files were.

## Already done

- **political_economy** (CTree_CEA) and **llm_distributions** (ECON-227) are migrated. Use them as the example.
- The COPY and rm lines were already taken out of **.dockerfile** on this branch, so there is nothing to do there.
- `freeze: true` is already set in the `execute:` block of **_quarto.yml**.

So the remaining work is the other 11 notebooks, one at a time.

## Before you start a notebook

You have to be able to actually run it. Open the `.qmd`, make sure its datasets are in the folder next to it and that the packages import. If you cannot run it (private data you do not have, a model you cannot download) then freeze will not work so you have to download the packages/data.

## The checklist (per notebook)

1. In **project/_quarto.yml**, find the `- "!docs/.../your_notebook.qmd"` line in the `render:` list and delete it. While the file is excluded Quarto will not write a freeze cache for it, so this has to come first.

2. From inside `project/`, run the full render. This is the same command the build runs:

   ```
   quarto render
   ```

   It will execute your newly un-excluded notebook once (you will see `Cell 1/12`, `Cell 2/12` ...) and write the cache to `project/_freeze/docs/.../notebook_name/`. The other excluded notebooks stay excluded so they do not run, and the two already migrated ones get read from their cache instead of running.

   Do not use `quarto render docs/.../your_notebook.qmd --to html` to make the committed cache. A single file render makes a cache with a different hash, and the next full render throws it out and runs the notebook again. Always make the committed cache with the full `quarto render`.

3. Check the cache is there and has outputs:

   ```
   project/_freeze/docs/.../your_notebook/execute-results/html.json
   project/_freeze/docs/.../your_notebook/figure-html/   (only if it makes matplotlib plots)
   ```

4. Run `quarto render` a second time. This time your notebook should NOT show `Cell x/y`. If it does not run, freeze is reading the cache and you are good. Open the page under `project/_site/docs/.../notebook_name.html` and check the plots are there and the code is highlighted.

5. Clean up the old way of doing it:
   - delete the old `project/docs/.../your_notebook.html`
   - delete the redirect stub `project/docs/.../your_notebook_stub.qmd`
   - in the index pages that pointed at the stub, change the `_stub.qmd` to the real `.qmd`. The stubs are referenced in `project/pages/index/`. For example llm_distributions is in `index_ECON227.qmd` and `all.qmd`, political_economy is in `all.qmd`.

6. Commit the whole `project/_freeze/docs/.../your_notebook/` folder together with the `_quarto.yml`, index and deletion changes.

## Things where I found issues so you can avoid them

- Un-exclude before you render. An excluded file gets no cache, no warning.
- The full `quarto render` is the one that counts. Single file renders run fine for a quick look but their cache does not survive the next full render.
- A full render moves rendered pages into `_site` and will delete loose `.html` sitting in the `docs/` tree, including ones you did not mean to touch (a presentation `.html` got removed when I did the pilot). Check `git status` after and `git checkout` anything that is not part of your notebook.
- If you change the notebook's *code* later, the cache is stale and you have to re-run the full render with the data to rebuild it. Same upkeep as re-rendering the `.html` was, just in a form the build can re-skin. Prose only edits are fine, the build handles those.
- Your local Quarto version does not have to match the build. The base image pins Quarto 1.4.557, the cache for the two pilots was made with 1.5.57, and a 1.4.557 render reads it straight from cache without re-running anything. So just use whatever Quarto you have.

## Remaining notebooks

- soci_415_network_analysis, kinmatrix, cbdb_dataset (SOCI-415, CBDB data, ~1.7 GB)
- amne_376_image_embedding (AMNE-376)
- soci_280_bert (SOCI-280)
- ocr_notebook (OCR)
- image_analysis (image_analysis)
- text_embeddings_workshop (hist_workshop)
- intro_to_cnn (intro_to_cnns)
- intro_to_convolution (intro_to_convolutions)
- intro_to_fundamental_ML (intro_to_deep_learning)
