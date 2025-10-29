# Fixes rendering of .ipynb headers and title

import json
import sys
import re
from pathlib import Path


def clean_notebook(ipynb_path):
    """Replace YAML frontmatter with formatted title, keep rest of cell."""
    
    try:
        # Read the notebook
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Check if notebook has cells
        if not nb.get('cells') or len(nb['cells']) == 0:
            print(f"Warning: {ipynb_path} has no cells")
            return
        
        # Get the first cell ONLY
        first_cell = nb['cells'][0]
        
        # Only process if it's a markdown cell
        if first_cell.get('cell_type') != 'markdown':
            print(f"Info: {ipynb_path} - first cell is not markdown, skipping")
            return
        
        # Get the source content
        source = first_cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        
        # Only process if it looks like YAML (starts with --- and has title:)
        if not (source.strip().startswith('---') and 'title:' in source):
            print(f"Info: {ipynb_path} - first cell doesn't have YAML, skipping")
            return
        
        print(f"Processing: {ipynb_path}")
        
        # Find where YAML ends (look for "name: python3" followed by "---")
        # This is the end marker for the YAML frontmatter
        yaml_end_pattern = r'name:\s*python3\s*\n---\n'
        yaml_end_match = re.search(yaml_end_pattern, source)
        
        if not yaml_end_match:
            print(f"Warning: Could not find YAML end marker (name: python3 + ---)")
            return
        
        # Split content: YAML part and everything after
        yaml_end_pos = yaml_end_match.end()
        yaml_part = source[:yaml_end_pos]
        rest_of_content = source[yaml_end_pos:]  # Everything after YAML
        
        # Extract title (handle quotes)
        title_match = re.search(r'title:\s*"([^"]+)"', yaml_part)
        if not title_match:
            title_match = re.search(r"title:\s*'([^']+)'", yaml_part)
        
        # Extract author
        author_match = re.search(r'author:\s*([^\n]+)', yaml_part)
        
        # Extract date (handle quotes)
        date_match = re.search(r"date:\s*'([^']+)'", yaml_part)
        if not date_match:
            date_match = re.search(r'date:\s*"([^"]+)"', yaml_part)
        
        # Get the values
        title = title_match.group(1).strip() if title_match else "Untitled"
        author = author_match.group(1).strip() if author_match else ""
        date = date_match.group(1).strip() if date_match else ""
        
        # Clean up author (remove HTML tags, underscores, asterisks)
        if author:
            author = re.sub(r'<br[^>]*>', ' ', author, flags=re.IGNORECASE)
            author = author.replace('_', '').replace('*', '')
            author = re.sub(r'\s+', ' ', author).strip()
        
        # Clean up date
        if date:
            date = date.strip("'\"")
        
        # Build new content: formatted title/author/date + rest of original content
        new_content = f"# {title}\n\n"
        
        if author and date:
            new_content += f"{author}, {date}\n\n"
        elif author:
            new_content += f"{author}\n\n"
        elif date:
            new_content += f"{date}\n\n"
        
        # Add back all the content that was AFTER the YAML
        new_content += rest_of_content
        
        # Replace the first cell with new content
        nb['cells'][0]['source'] = new_content
        
        # Write back the notebook
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Cleaned successfully")
        print(f"    Title: {title}")
        print(f"    Author: {author}")
        print(f"    Date: {date}")
        print(f"    Preserved {len(rest_of_content)} characters of content after YAML")
        
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
