# Converts Quarto callout syntax to HTML in .ipynb markdown cells
import json
import re
import sys
import glob
from pathlib import Path

CALLOUT_STYLES = {
    'callout-tip':       {'border': '#4e9af1', 'background': '#f0f7ff', 'emoji': '💡', 'label': 'Tip'},
    'callout-note':      {'border': '#4e9af1', 'background': '#f0f7ff', 'emoji': '📝', 'label': 'Note'},
    'callout-warning':   {'border': '#f0ad4e', 'background': '#fff8f0', 'emoji': '⚠️',  'label': 'Warning'},
    'callout-important': {'border': '#d9534f', 'background': '#fff0f0', 'emoji': '❗', 'label': 'Important'},
    'callout-caution':   {'border': '#f0ad4e', 'background': '#fff8f0', 'emoji': '🔔', 'label': 'Caution'},
}

_TYPES = '|'.join(CALLOUT_STYLES)

CALLOUT_RE = re.compile(
    rf':::+\s*(?:\{{[^}}]*\.({_TYPES})[^}}]*\}}|({_TYPES}))[ \t]*\n'
    r'(.*?)'
    r'\n:::+[ \t]*(?=\n|$)',
    re.DOTALL,
)

def _build_html(callout_type: str, content: str) -> str:
    style = CALLOUT_STYLES[callout_type]
    content = content.strip()
    # >>> NEW: Use a table with legacy `bgcolor` attributes instead of CSS
    # >>> `style`, since Google Colab's HTML sanitizer strips `style` (and
    # >>> thus any background/border/padding from a styled <div>) but still
    # >>> permits `bgcolor`/`width`/`border`/`cellpadding` on tables.
    # >>> Jupyter renders this identically to the styled div. Blank lines
    # >>> are still needed around the body so markdown still renders inside
    # >>> the raw HTML block (CommonMark closes the HTML block at the first
    # >>> blank line).
    return (
        f'<table border="0" cellpadding="0" cellspacing="0" width="100%">\n'
        f'<tr>\n'
        f'<td bgcolor="{style["border"]}" width="4"></td>\n'
        f'<td bgcolor="{style["background"]}">\n'
        f'<table border="0" cellpadding="12" cellspacing="0" width="100%">\n'
        f'<tr><td bgcolor="{style["background"]}">\n\n'
        f'<strong>{style["emoji"]} {style["label"]}:</strong>\n\n'
        f'{content}\n\n'
        f'</td></tr>\n'
        f'</table>\n'
        f'</td>\n'
        f'</tr>\n'
        f'</table>'
    )

def convert_callouts_in_notebook(ipynb_path: str):
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

            if not CALLOUT_RE.search(source_text):
                continue

            new_text = CALLOUT_RE.sub(
                lambda m: _build_html(m.group(1) or m.group(2), m.group(3)),
                source_text
            )

            if new_text != source_text:
                nb['cells'][i]['source'] = new_text.splitlines(keepends=True)
                changed = True
                print(f"    Converted callout in cell {i}")

        if changed:
            with open(ipynb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2, ensure_ascii=False)
            print(f"Saved: {ipynb_path}")
        else:
            print(f"No callouts found: {ipynb_path}")

    except Exception as e:
        print(f"Error processing {ipynb_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_callouts.py <notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    if not Path(notebook_path).exists():
        print(f"Error: File not found: {notebook_path}")
        sys.exit(1)

    convert_callouts_in_notebook(notebook_path)
    