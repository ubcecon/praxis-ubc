# Fixes rendering of .ipynb headers and title

import json
import sys
import re
from pathlib import Path

try:
    import yaml  # Optional; falls back to regex if unavailable
except Exception:
    yaml = None


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
        
        # Look for a YAML frontmatter block delimited by lines with only '---'
        yaml_block_pattern = r'^\s*---\s*\n(.*?)\n^\s*---\s*\n?'
        yaml_block_match = re.search(yaml_block_pattern, source, flags=re.S | re.M)
        if not yaml_block_match:
            print(f"Info: {ipynb_path} - first cell doesn't have YAML frontmatter, skipping")
            return
        
        print(f"Processing: {ipynb_path}")
        
        yaml_part = yaml_block_match.group(1)
        rest_of_content = source[yaml_block_match.end():]
        
        # Parse YAML properly to handle multi-line values and complex structures
        yaml_data = {}
        if yaml is not None:
            try:
                parsed_yaml = yaml.safe_load(yaml_part) or {}
                if isinstance(parsed_yaml, dict):
                    yaml_data = parsed_yaml
            except yaml.YAMLError:
                print(f"Warning: {ipynb_path} - invalid YAML, falling back to regex parsing")
            except Exception:
                print(f"Warning: {ipynb_path} - unexpected error parsing YAML, falling back to regex parsing")
        else:
            print(f"Info: {ipynb_path} - PyYAML not installed, using regex parsing")
        
        # Extract values with fallback to regex if YAML parsing fails
        def get_field(field_name):
            if yaml_data and field_name in yaml_data:
                return str(yaml_data[field_name]).strip()
            
            # Fallback: regex for simple field: value patterns
            pat = rf'^\s*{re.escape(field_name)}:\s*(?:"([^"]*)"|\'([^\']*)\'|([^\n]*))\s*$'
            m = re.search(pat, yaml_part, flags=re.M)
            if m:
                return (m.group(1) or m.group(2) or m.group(3) or "").strip()
            return ""
        
        # Extract values
        title = get_field('title') or "Untitled"
        author = get_field('author')
        date = get_field('date')
        
        # Clean up title (remove quotes)
        title = title.strip('"\'')
        
        # Clean up author (remove HTML tags, but preserve formatting)
        if author:
            author = author.strip('"\'')
            # Convert <br> tags to spaces but preserve other formatting
            author = re.sub(r'<br[^>]*>', ' ', author, flags=re.IGNORECASE)
            # Clean up extra whitespace
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
        
        # Replace the first cell with new content; store as list of lines to match typical nbformat
        nb['cells'][0]['source'] = new_content.splitlines(keepends=True)
        
        # Write back the notebook
        with open(ipynb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        
        print(f"    Cleaned successfully")
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
