#Fix .ipynb titles from comet_main.yml render
import json
import sys
import re
from pathlib import Path


def clean_notebook(ipynb_path):
    """Clean the first cell of a Jupyter notebook if it contains YAML frontmatter."""
    
    try:
        # Read the notebook
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Check if notebook has cells
        if not nb.get('cells'):
            print(f"Warning: {ipynb_path} has no cells")
            return
        
        # Get the first cell
        first_cell = nb['cells'][0]
        
        # Check if first cell is markdown (YAML frontmatter appears as 'markdown' cell type)
        if first_cell.get('cell_type') != 'markdown':
            print(f"Info: {ipynb_path} - first cell is not 'markdown' type, skipping")
            return
        
        # Get the source content (can be a list or string)
        source = first_cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        
        # Check if it contains YAML frontmatter (starts with --- and has title:)
        if not (source.strip().startswith('---') and 'title:' in source):
            print(f"Info: {ipynb_path} - first cell doesn't look like YAML frontmatter")
            return
        
        print(f"Processing: {ipynb_path}")
        
        # Extract title, author, date using regex
        # Handle quoted titles
        title_match = re.search(r'title:\s*"([^"]+)"', source)
        if not title_match:
            title_match = re.search(r"title:\s*'([^']+)'", source)
        if not title_match:
            title_match = re.search(r'title:\s*([^\n]+)', source)
        
        # Extract author (may have HTML tags)
        author_match = re.search(r'author:\s*([^\n]+)', source)
        
        # Extract date (handle quoted dates)
        date_match = re.search(r"date:\s*'([^']+)'", source)
        if not date_match:
            date_match = re.search(r'date:\s*"([^"]+)"', source)
        if not date_match:
            date_match = re.search(r'date:\s*([^\n]+)', source)
        
        # Extract and clean values
        title = title_match.group(1).strip() if title_match else None
        author = author_match.group(1).strip() if author_match else None
        date = date_match.group(1).strip() if date_match else None
        
        # Clean up author (remove HTML tags, markdown formatting)
        if author:
            # Remove <br>, <br/>, <br />, etc.
            author = re.sub(r'<br[^>]*>', ' ', author, flags=re.IGNORECASE)
            # Remove underscores (markdown italics)
            author = author.replace('_', '')
            # Remove asterisks (markdown bold)
            author = author.replace('*', '')
            # Normalize whitespace
            author = re.sub(r'\s+', ' ', author).strip()
        
        # Clean up date (remove quotes if any remain)
        if date:
            date = date.strip("'\"")
        
        # Build new clean YAML with ONLY title, author, date
        new_lines = [
            '---\n',
        ]
        
        if title:
            new_lines.append(f'title: "{title}"\n')
        
        if author:
            new_lines.append(f'author: "{author}"\n')
        
        if date:
            new_lines.append(f'date: "{date}"\n')
        
        new_lines.append('---\n')
        
        # Replace the first cell's source
        nb['cells'][0]['source'] = new_lines
        
        # Write the modified notebook back
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Cleaned successfully")
        print(f"    Title: {title}")
        print(f"    Author: {author}")
        print(f"    Date: {date}")
        
    except json.JSONDecodeError as e:
        print(f"Error: {ipynb_path} is not valid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing {ipynb_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python clean-ipynb-titles.py <notebook.ipynb>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    
    if not Path(notebook_path).exists():
        print(f"Error: File not found: {notebook_path}")
        sys.exit(1)
    
    clean_notebook(notebook_path)
