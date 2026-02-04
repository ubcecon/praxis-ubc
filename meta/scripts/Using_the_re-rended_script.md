# Re-render Self-Contained QMD Files

This script automates the process of re-rendering `.qmd` files that need to be self-contained HTML files for the prAxIs website.

## Background

Some notebooks in prAxIs require packages and datasets that are not available in the Docker build environment. These files are:
- Excluded from the Docker build (listed in `_quarto.yml` with `!` prefix)
- Rendered locally as self-contained HTML files
- The HTML is copied alongside the source `.qmd` for deployment

For more details, see `documentation/website_and_repository.qmd`, Section 6.2.

## Prerequisites

1. **Quarto** must be installed and available in your PATH
   - Download from: https://quarto.org/docs/get-started/
   - Verify installation: `quarto --version`

2. **Python 3.9+** is required

3. **Required packages** for the notebooks you're rendering must be installed locally
   - Each notebook may have different requirements (check the notebook headers)

## Usage

### Basic Usage

From any directory, run:

```bash
# Render all excluded QMD files
python D:\GitHub\praxis-ubc\meta\scripts\re_render_qmd_files.py

# Or navigate to the repository first
cd D:\GitHub\praxis-ubc
python meta\scripts\re_render_qmd_files.py
```

### Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Show what would be done without actually rendering |
| `--file FILTER` | Only render files matching the filter (partial match) |
| `--skip-verify` | Skip verification of required headers |
| `--verbose, -v` | Show detailed output |

### Examples

```bash
# Preview what files would be rendered
python re_render_qmd_files.py --dry-run

# Render only SOCI-415 files
python re_render_qmd_files.py --file soci_415

# Render only the network analysis notebook
python re_render_qmd_files.py --file network_analysis

# Render with verbose output
python re_render_qmd_files.py --verbose

# Skip header verification (use with caution)
python re_render_qmd_files.py --skip-verify
```

## How It Works

1. **Discovery**: Parses `project/_quarto.yml` to find files excluded from the build (lines with `!` prefix)

2. **Verification**: Checks that each file has the required self-contained headers:
   ```yaml
   execute:
     eval: true
     echo: true
     output: true
   format:
     html:
       embed-resources: true
       self-contained-math: true
   ```

3. **Rendering**: For each file:
   - Runs `quarto render FILENAME.qmd` from the project directory
   - The HTML is generated in `project/_site/docs/CLASS/...`
   - Copies the HTML to `project/docs/CLASS/...` (same directory as source)

4. **Reporting**: Provides a summary showing successful and failed renders

## Files Currently Configured for Re-rendering

As of 2025-01-24, these files are excluded from Docker build and need local rendering:

| File | Course |
|------|--------|
| `soci_415_network_analysis.qmd` | SOCI-415 |
| `kinmatrix.qmd` | SOCI-415 |
| `cbdb_dataset.qmd` | SOCI-415 |
| `llm_distributions.qmd` | ECON-227 |
| `text_embeddings_workshop.qmd` | hist_workshop |
| `intro_to_cnn.qmd` | intro_to_cnns |
| `intro_to_convolution.qmd` | intro_to_convolutions |
| `intro_to_fundamental_ML.qmd` | intro_to_deep_learning |
| `amne_376_image_embedding.qmd` | AMNE-376 |
| `soci_280_bert.qmd` | SOCI-280 |

## Troubleshooting

### "Quarto is not installed or not in PATH"
Install Quarto from https://quarto.org/docs/get-started/

### Render fails with package errors
Make sure you have all required Python/R packages installed locally. Check the notebook's first cells for package requirements.

### HTML not found after rendering
- Ensure you're running from the repository root
- Check the Quarto output for errors
- Verify the file has correct headers

### Missing required headers warning
The file needs the self-contained headers shown above. Either:
- Add the headers to the file, or
- Use `--skip-verify` if you're sure the file is correct

## Adding New Files

To add a new file for local rendering:

1. Add self-contained headers to the `.qmd` file (see above)

2. Add exclusion to `project/_quarto.yml`:
   ```yaml
   render:
     - "!docs/YOUR-CLASS/your_file.qmd"
   ```

3. Add to `meta/building/.dockerfile`:
   ```dockerfile
   RUN rm -f ./docs/YOUR-CLASS/your_file.qmd
   COPY ./project/docs/YOUR-CLASS/your_file.html /app/output/docs/YOUR-CLASS/
   ```

4. Create a stub file `your_file_stub.qmd`:
   ```yaml
   ---
   title: Your Title
   # ... other metadata ...
   ---

   <meta http-equiv="refresh" content="0; url=your_file.html">

   If you are not redirected automatically, [click here](your_file.html).
   ```

5. Update the index page to use the stub instead of the original

6. Run this script to render the HTML locally
