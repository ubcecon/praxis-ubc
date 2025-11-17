# Fixes rendering of .ipynb headers and title using regex-only parsing
import json
import sys
import re
from pathlib import Path

# Precompiled regexes
YAML_BLOCK_RE = re.compile(
    r'^(?:\ufeff)?\s*---[ \t]*\r?\n'          # opening delimiter
    r'(.*?)'                                   # YAML content (non-greedy)
    r'\r?\n^\s*(?:---|\.{3}|-{3,})[ \t]*\r?\n?',  # closing delimiter: --- or ... or 3+ dashes
    flags=re.S | re.M,
)

def _get_field(yaml_text: str, name: str) -> str:
    # Capture the rest of the line after "name:" allowing quotes or unquoted
    # Works for single-line scalar values.
    pat = re.compile(rf'^\s*{re.escape(name)}\s*:\s*(.*?)\s*$', flags=re.M)
    m = pat.search(yaml_text)
    if not m:
        return ""
    val = m.group(1).strip()
    # Remove matching surrounding quotes if present
    if (len(val) >= 2) and ((val[0] == val[-1]) and val[0] in ('"', "'")):
        val = val[1:-1]
    return val.strip()

def _clean_author(author: str) -> str:
    if not author:
        return ""
    # Normalize <br> tags to real newlines, keep underscores/emphasis intact
    author = re.sub(r'<br\s*/?>', '\n', author, flags=re.IGNORECASE)
    # Normalize CRLF
    author = author.replace('\r\n', '\n').replace('\r', '\n')
    # Trim spaces on each line but preserve line breaks
    lines = [ln.strip() for ln in author.split('\n')]
    # Remove empty leading/trailing lines
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return '\n'.join(lines)

def clean_notebook(ipynb_path):
    """Replace YAML frontmatter with formatted title (H1) and author/date; keep rest of first cell."""
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        if not nb.get('cells'):
            print(f"Warning: {ipynb_path} has no cells")
            return

        first_cell = nb['cells'][0]
        if first_cell.get('cell_type') != 'markdown':
            print(f"Info: {ipynb_path} - first cell is not markdown, skipping")
            return

        source = first_cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = str(source)

        # If already starts with a markdown H1 and not YAML, skip to avoid duplicating
        if re.match(r'^\s*#\s+\S', source_text) and not YAML_BLOCK_RE.search(source_text):
            print(f"Info: {ipynb_path} - already has a title, skipping")
            return

        m = YAML_BLOCK_RE.search(source_text)
        if not m:
            print(f"Info: {ipynb_path} - first cell doesn't have YAML frontmatter, skipping")
            return

        print(f"Processing: {ipynb_path}")

        yaml_part = m.group(1)
        rest_of_content = source_text[m.end():]

        # Extract fields
        title = _get_field(yaml_part, 'title') or "Untitled"
        author = _get_field(yaml_part, 'author')
        date = _get_field(yaml_part, 'date')

        # Clean fields
        title = title.strip()
        author = _clean_author(author)
        date = date.strip() if date else ""

        # Build new content
        new_parts = []
        new_parts.append(f"# {title}\n\n")
        if author and date:
            if '\n' in author:
                new_parts.append(f"{author}\n\n{date}\n\n")
            else:
                new_parts.append(f"{author}, {date}\n\n")
        elif author:
            new_parts.append(f"{author}\n\n")
        elif date:
            new_parts.append(f"{date}\n\n")
        new_parts.append(rest_of_content)
        new_content = ''.join(new_parts)

        # Only write if changed
        if new_content == source_text:
            print("    No changes needed")
            return

        nb['cells'][0]['source'] = new_content.splitlines(keepends=True)

        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)

        print("    Cleaned successfully")
        print(f"    Title: {title}")
        print(f"    Author: {author.replace(chr(10), ' / ')}")
        print(f"    Date: {date}")
        print(f"    Preserved {len(rest_of_content)} characters after YAML")
    except Exception as e:
        print(f"Error processing {ipynb_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_test_notebook(target_path: Path):
    """Create a test notebook with YAML frontmatter for local testing only."""
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "title: \"A Study of Richter's Kouroi Through Image Embedding\"\n",
                    "author: prAxIs UBC Team <br> _Kaiyan Zhang, Yash Mali, Krishaant Pathman_\n",
                    "date: 2025-07-25\n",
                    "description: Using examples from Richter's *Kouroi*, this notebook introduces computer vision and image embeddings.\n",
                    "categories:\n",
                    "  - AMNE 376\n",
                    "  - Python\n",
                    "  - image embeddings\n",
                    "  - convolutions\n",
                    "  - neural networks\n",
                    "format:\n",
                    "  html:\n",
                    "    code-fold: true\n",
                    "    code-summary: \"Show the code\"\n",
                    "  ipynb:\n",
                    "    jupyter:\n",
                    "      kernelspec:\n",
                    "        display_name: Python\n",
                    "        language: python3\n",
                    "        name: python3\n",
                    "---\n\n",
                    "This is the introduction paragraph that should remain after the YAML is removed.\n",
                    "It should appear after the generated title block.\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Hello from a code cell')\n"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Test notebook created at: {target_path}")

if __name__ == '__main__':
    # Local test handle. Not used in deployment unless explicitly invoked.
    if len(sys.argv) == 2 and sys.argv[1] == '--test':
        script_dir = Path(__file__).parent
        test_path = script_dir / "test_clean_ipynb_titles.ipynb"
        create_test_notebook(test_path)
        clean_notebook(str(test_path))
        # Optionally show a preview of the cleaned first cell
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            first_cell_src = nb['cells'][0].get('source', [])
            preview = ''.join(first_cell_src)[:400] if isinstance(first_cell_src, list) else str(first_cell_src)[:400]
            print("\nPreview of cleaned first cell:\n")
            print(preview)
        except Exception as e:
            print(f"Could not preview test notebook: {e}")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python clean_ipynb_titles.py <notebook.ipynb>")
        print("  python clean_ipynb_titles.py --test   # create and clean a test notebook (local only)")
        sys.exit(1)

    notebook_path = sys.argv[1]
    if not Path(notebook_path).exists():
        print(f"Error: File not found: {notebook_path}")
        sys.exit(1)

    clean_notebook(notebook_path)

