# clean-ipynb-titles.py
import json
import sys
import re
from pathlib import Path

def clean_notebook(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Check if first cell is the raw YAML
    if nb['cells'] and nb['cells'][0].get('cell_type') == 'raw':
        first_cell = nb['cells'][0]
        source = ''.join(first_cell.get('source', []))
        
        # Check if it's YAML frontmatter
        if source.strip().startswith('---') and 'title:' in source:
            # Extract title, author, date
            title_match = re.search(r'title:\s*["\']?([^"\'\n]+)["\']?', source)
            author_match = re.search(r'author:\s*([^\n]+)', source)
            date_match = re.search(r'date:\s*["\']?([^"\'\n]+)["\']?', source)
            
            title = title_match.group(1).strip() if title_match else None
            author = author_match.group(1).strip() if author_match else None
            date = date_match.group(1).strip() if date_match else None
            
            # Clean author
            if author:
                author = re.sub(r'<br[^>]*>', ' ', author)
                author = re.sub(r'<BR[^>]*>', ' ', author)
                author = author.replace('_', '').replace('*', '')
                author = re.sub(r'\s+', ' ', author).strip()
            
            # Clean date
            if date:
                date = date.strip("'\"")
            
            # Build new clean YAML
            new_lines = ['---\n']
            if title:
                new_lines.append(f'title: "{title}"\n')
            if author:
                new_lines.append(f'author: "{author}"\n')
            if date:
                new_lines.append(f'date: "{date}"\n')
            new_lines.append('---')
            
            # Replace the cell
            nb['cells'][0]['source'] = new_lines
    
    # Write back
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        clean_notebook(sys.argv[1])
