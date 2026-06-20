# Quarto's figure and column syntax doesn't render once a notebook is opened in
# Jupyter or Colab, so the images go missing. This rewrites those blocks as plain
# HTML tables that show up anywhere.
# Usage: python clean_ipynb_images.py <notebook.ipynb>
# Runs in comet_main.yml right after convert_ipynb_callouts.py
import json
import re
import sys

# Pattern 1: figure blocks like ::: {#fig-x layout-ncol=2} ... :::
# Group 1 is the attributes, group 2 is everything inside.
FIGURE_BLOCK_RE = re.compile(
    r'^:::+\s*\{([^}]*)\}\s*\n((?:[^\n]*\n)*?)^:::+[ \t]*$',
    re.MULTILINE,
)

# Pulls the ![alt](src) images out of a block. The "nested" version covers the
# awkward case where the alt text itself has a [link](...) inside it.
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]*)\)')
_IMAGE_RE_NESTED = re.compile(r'!\[(.*)\]\(([^)]*)\)')

def _find_images(content):
    results = []
    for line in content.split('\n'):
        if line.count('![') == 1:
            m = _IMAGE_RE_NESTED.search(line)
            if m:
                results.append(m.groups())
                continue
        results.extend(_IMAGE_RE.findall(line))
    return results

_COLUMNS_OPEN_RE = re.compile(r'^(:::+)\s*\{\.columns[^}]*\}\s*$')
_FENCE_RE = re.compile(r'^(:::+)(.*)$')

_COLUMN_SEP = '\x00COLSEP\x00'

def _merge_all_columns_blocks(text: str) -> str:
    # Process each .columns block separately using fence-depth tracking.
    # Prevents multiple column blocks from being merged by a greedy match.
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        m = _COLUMNS_OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue

        start = i
        depth = 1
        j = i + 1
        while j < len(lines) and depth > 0:
            if _FENCE_RE.match(lines[j]):
                depth += 1 if '{' in lines[j] else -1
            j += 1
        merged = _merge_columns_block('\n'.join(lines[start:j]))
        out.append(merged)
        i = j
    return '\n'.join(out)

def _merge_columns_block(block_text: str) -> str:
    # Convert a .columns block into a figure block and preserve
    # column boundaries with a sentinel for later processing.
    lines = block_text.split('\n')
    # first line is the .columns opener, last line is its matching closer
    inner_lines = lines[1:-1]

    columns = []
    k = 0
    while k < len(inner_lines):
        cm = _FENCE_RE.match(inner_lines[k])
        if cm and '{' in inner_lines[k] and '.column' in inner_lines[k] and '.columns' not in inner_lines[k]:
            depth = 1
            col_start = k + 1
            m2 = k + 1
            while m2 < len(inner_lines) and depth > 0:
                if _FENCE_RE.match(inner_lines[m2]):
                    depth += 1 if '{' in inner_lines[m2] else -1
                m2 += 1
            columns.append('\n'.join(inner_lines[col_start:m2 - 1]).strip())
            k = m2
        else:
            k += 1

    if not columns:
        return block_text
    ncol = len(columns)
    body = f'\n{_COLUMN_SEP}\n'.join(columns)
    return f'::: {{layout-ncol={ncol}}}\n{body}\n:::'


def _parse_ncol(attrs: str) -> int:
    m = re.search(r'layout-ncol=(\d+)', attrs)
    return int(m.group(1)) if m else 1

def _split_columns(content: str, ncol: int) -> list[str]:
    # Recover original column chunks using the sentinel inserted
    # by _merge_columns_block; otherwise treat content as one column.
    if _COLUMN_SEP in content:
        return content.split(_COLUMN_SEP)
    return [content]


def _build_figure_html(attrs: str, content: str) -> str:
    # Convert a figure block into an HTML table of images.
    ncol = _parse_ncol(attrs)
    if ncol == 1 and re.search(r'^\.column\b', attrs.strip()):
        return content.strip()

    images = _find_images(content)   # [(alt, src), ...]

 # Split the content into per-column chunks so we can detect text-only columns.
    col_chunks = _split_columns(content, ncol)
    text_only = [c for c in col_chunks if not _find_images(c) and c.strip()]

    if not images:
        # Nothing to convert here, so hand the text back as-is and lose nothing.
        return content.replace(_COLUMN_SEP, '').strip()
        # Marker used to preserve exact column boundaries after flattening.

    pct = int(100 / ncol)
    if ncol > 1 and len(images) == ncol and not text_only:
        headings = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        cells = ''.join(
            f'<td align="center" width="{pct}%">'
            + (f'<b>{headings[idx]}</b><br>' if idx < len(headings) else '')
            + f'<img src="{src}" alt="{alt}" style="max-width:100%">'
            f'</td>'
            for idx, (alt, src) in enumerate(images)
        )
    else:
        cells = ''.join(
            f'<td align="center" width="{pct}%">'
            f'<img src="{src}" alt="{alt}" style="max-width:100%">'
            f'</td>'
            for alt, src in images
        )

    table = (
        f'<table border="0" cellpadding="8" cellspacing="0" width="100%">\n'
        f'<tr>{cells}</tr>\n'
        f'</table>'
    )
    if text_only:
        trailing = '\n\n'.join(chunk.strip() for chunk in text_only)
        return f'{table}\n\n{trailing}'
    return table
    
# Pattern 2: images with trailing Quarto attributes like ![alt](src){width="50%"}.
# Jupyter prints the {...} as literal text, so we just drop it.
IMAGE_ATTR_RE = re.compile(r'(!\[[^\]]*\]\([^)]*\))\{[^}]*\}')

def clean_images_in_notebook(ipynb_path: str):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'markdown':
            continue

        source_text = ''.join(cell['source'])

        # Run the chain a step at a time and note which ones actually fired, so
        # the log says what changed. Each step is a no-op when its syntax isn't
        # there, so a cell with none of it falls through untouched.
        changes = []
        new_text = _merge_all_columns_blocks(source_text)
        if new_text != source_text:
            changes.append("columns blocks")

        stepped = FIGURE_BLOCK_RE.sub(
            lambda m: _build_figure_html(m.group(1), m.group(2)), new_text
        )
        if stepped != new_text:
            changes.append("figure blocks")
        new_text = stepped

        stepped = IMAGE_ATTR_RE.sub(r'\1', new_text)
        if stepped != new_text:
            changes.append("image attrs")
        new_text = stepped

        if not changes:
            continue

        # Only tidy up blank lines in cells we actually changed.
        new_text = re.sub(r'\n{3,}', '\n\n', new_text)
        cell['source'] = new_text.splitlines(keepends=True)
        changed = True
        print(f"    Converted {', '.join(changes)} in cell {i}")

    if changed:
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Saved: {ipynb_path}")
    else:
        print(f"No image syntax found: {ipynb_path}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean_ipynb_images.py <notebook.ipynb>")
        sys.exit(1)
    clean_images_in_notebook(sys.argv[1])
