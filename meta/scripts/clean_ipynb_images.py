# comet_main.yml runs it once per file, after convert_ipynb_callouts.py.
import json
 
import re
 
import sys
 
from pathlib import Path
 
 
 
# --- Pattern 1: figure div blocks -----------------------------------------
 
# Matches a complete ::: {attrs} ... ::: block.
 
# Group 1 = attribute string, group 2 = inner content.
 
# so the regex stops at the closing ::: instead of overshooting.
 
FIGURE_BLOCK_RE = re.compile(
 
    r'^:::+\s*\{([^}]*)\}\s*\n((?:[^\n]*\n)*?)^:::+[ \t]*$',
 
    re.MULTILINE,
 
)
 
 
 
# Finds individual markdown images inside a figure block.
 
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\(([^)]*)\)')
_IMAGE_RE_NESTED = re.compile(r'!\[(.*)\]\(([^)]*)\)')
 
 
def _find_images(content):
    results = []
    for line in content.split(chr(10)):
        if line.count('![') == 1:
            m = _IMAGE_RE_NESTED.search(line)
            if m:
                results.append(m.groups())
                continue
        results.extend(_IMAGE_RE.findall(line))
    return results
 
 
_COLUMNS_OPEN_RE = re.compile(r'^(:::+)\s*\{\.columns[^}]*\}\s*$')
_FENCE_RE = re.compile(r'^(:::+)(.*)$')
 
 
def _merge_all_columns_blocks(text: str) -> str:
    """Find each ::: {.columns} ... ::: block by tracking fence depth (an
    opening fence has {attrs}, a bare ::: closes one level), so each
    .columns wrapper is merged independently instead of one regex
    overshooting from the first .columns to the LAST ::: in the cell."""
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
            fm = _FENCE_RE.match(lines[j])
            if fm:
                if '{' in lines[j]:
                    depth += 1
                else:
                    depth -= 1
            j += 1
        block_lines = lines[start:j]
        merged = _merge_columns_block('\n'.join(block_lines))
        out.append(merged)
        i = j
    return '\n'.join(out)
 
 
def _merge_columns_block(block_text: str) -> str:
    """Turn one ::: {.columns} ... ::: block (with nested ::: {.column}
    blocks at depth 1) into a synthetic {layout-ncol=N} figure block."""
    lines = block_text.split('\n')
    # lines[0] is the .columns opener, lines[-1] is its matching closer.
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
                fm2 = _FENCE_RE.match(inner_lines[m2])
                if fm2:
                    if '{' in inner_lines[m2]:
                        depth += 1
                    else:
                        depth -= 1
                m2 += 1
            columns.append('\n'.join(inner_lines[col_start:m2 - 1]).strip())
            k = m2
        else:
            k += 1
 
    if not columns:
        return block_text
    ncol = len(columns)
    body = '\n'.join(columns)
    return f'::: {{layout-ncol={ncol}}}\n{body}\n:::'
 
 
 
def _parse_ncol(attrs: str) -> int:
 
    m = re.search(r'layout-ncol=(\d+)', attrs)
 
    return int(m.group(1)) if m else 1
 
 
 
 
 
def _build_figure_html(attrs: str, content: str) -> str:
 
    """Render a Quarto figure block as an HTML table of images."""
 
    ncol   = _parse_ncol(attrs)
 
    if ncol == 1 and re.search(r'^\.column\b', attrs.strip()):
        return content.strip()    
 
    images = _find_images(content)   # [(alt, src), ...]
 
 
 
    if not images:
 
        # No images found — return inner content as-is so nothing is lost.
 
        return content.strip()
 
 
 
    pct   = int(100 / ncol)
 
    if ncol > 1 and len(images) == ncol:
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
 
    # bgcolor keeps compatibility with Colab (strips style= but keeps bgcolor).
 
    return (
 
        f'<table border="0" cellpadding="8" cellspacing="0" width="100%">\n'
 
        f'<tr>{cells}</tr>\n'
 
        f'</table>'
 
    )
 
 
 
# --- Pattern 2: inline image attribute blocks ------------------------------
  
# Matches ![alt](src){key="value" ...} and strips the {...} block.
 
IMAGE_ATTR_RE = re.compile(r'(!\[[^\]]*\]\([^)]*\))\{[^}]*\}')
 
 
 
 
# --- Main ------------------------------------------------------------------
 
 
 
def clean_images_in_notebook(ipynb_path: str):
 
    try:
 
        with open(ipynb_path, 'r', encoding='utf-8') as f:
 
            nb = json.load(f)
 
 
 
        if not nb.get('cells'):
 
            print(f"Warning: {ipynb_path} has no cells")
 
            return
 
 
 
        changed = False
 
 
 
        for i, cell in enumerate(nb['cells']):
 
            if cell.get('cell_type') != 'markdown':
 
                continue
 
 
 
            source = cell.get('source', [])
 
            source_text = ''.join(source) if isinstance(source, list) else str(source)
 
            has_columns_block = '.columns' in source_text
 
            working_text = source_text
            if has_columns_block:
                working_text = _merge_all_columns_blocks(working_text)
 
            has_figure_block = FIGURE_BLOCK_RE.search(working_text)
            has_image_attrs  = IMAGE_ATTR_RE.search(working_text) 
 
            if not has_figure_block and not has_image_attrs:
 
                continue
 
 
 
            new_text = working_text
 
            if has_figure_block:
 
                new_text = FIGURE_BLOCK_RE.sub(
 
                    lambda m: _build_figure_html(m.group(1), m.group(2)),
 
                    new_text,
 
                )
 
            if has_image_attrs:
 
                new_text = IMAGE_ATTR_RE.sub(r'\1', new_text)
 
            new_text = re.sub(r'\n{3,}', '\n\n', new_text)
 
 
 
            if new_text != source_text:
 
                nb['cells'][i]['source'] = new_text.splitlines(keepends=True)
 
                changed = True
 
                changes = []
 
                if has_columns_block: changes.append("columns blocks")
 
                if has_figure_block: changes.append("figure blocks")
 
                if has_image_attrs:  changes.append("image attrs")
 
                print(f"    Converted {', '.join(changes)} in cell {i}")
 
 
 
        if changed:
 
            with open(ipynb_path, 'w', encoding='utf-8') as f:
 
                json.dump(nb, f, indent=2, ensure_ascii=False)
 
            print(f"Saved: {ipynb_path}")
 
        else:
 
            print(f"No image syntax found: {ipynb_path}")
 
 
 
    except Exception as e:
 
        print(f"Error processing {ipynb_path}: {e}")
 
        import traceback
 
        traceback.print_exc()
 
        sys.exit(1)
 
 
 
 
 
if __name__ == '__main__':
 
    if len(sys.argv) < 2:
 
        print("Usage:")
 
        print("  python clean_ipynb_images.py <notebook.ipynb>")
 
        sys.exit(1)
 
 
 
    notebook_path = sys.argv[1]
 
    if not Path(notebook_path).exists():
 
        print(f"Error: File not found: {notebook_path}")
 
        sys.exit(1)
 
 
 
    clean_images_in_notebook(notebook_path)