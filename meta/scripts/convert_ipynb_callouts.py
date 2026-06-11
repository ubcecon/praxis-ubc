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

CALLOUT_RE = re.compile(
    r':::\s*\{[^}]*\.(callout-tip|callout-note|callout-warning|callout-important|callout-caution)[^}]*\}\s*\n'
    r'(.*?)'
    r'\n:::\s*(?=\n|$)',
    re.DOTALL,
)

def _build_html(callout_type: str, content: str) -> str:
    style = CALLOUT_STYLES[callout_type]
    content = content.strip()
    return (
        f'<div style="border-left: 4px solid {style["border"]}; '
        f'background: {style["background"]}; '
        f'padding: 12px 16px; border-radius: 4px; margin: 12px 0;">\n'
        f'<strong>{style["emoji"]} {style["label"]}:</strong><br>\n'
        f'{content}\n'
        f'</div>'
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
                lambda m: _build_html(m.group(1), m.group(2)),
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
    