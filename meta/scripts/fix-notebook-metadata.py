import json
import os
import glob
import yaml
import re


def parse_yaml_frontmatter(source_lines):
    """Extract YAML content and parse it"""
    yaml_content = ''.join(source_lines)
    yaml_content = re.sub(r'^---\n?', '', yaml_content)
    yaml_content = re.sub(r'\n?---\s*$', '', yaml_content)
    
    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        print(f"Warning: Could not parse YAML: {e}")
        return None


def create_formatted_header_cell(yaml_data):
    """Create a nicely formatted markdown cell from YAML data"""
    if not yaml_data:
        return None
    
    lines = []
    
    # Add title as H1
    if 'title' in yaml_data:
        title = yaml_data['title']
        lines.append(f"# {title}\n")
    
    # Add author(s) with proper formatting
    if 'author' in yaml_data:
        author = yaml_data['author']
        # Convert HTML <br> to markdown line breaks
        author = author.replace('<br>', '  \n')
        # Convert underscores to markdown italics
        author = author.replace('_', '*')
        lines.append(f"{author}  \n")
    
    # Add date
    if 'date' in yaml_data:
        date = yaml_data['date']
        lines.append(f"{date}\n")
    
    if not lines:
        return None
    
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def fix_notebook_display(notebook_path):
    """
    Fix notebook to have:
    1. Raw cell with YAML (for Quarto)
    2. Markdown cell with formatted title/author/date (for display)
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells') or len(notebook['cells']) == 0:
            return False
        
        first_cell = notebook['cells'][0]
        
        # Get the source content
        source = first_cell.get('source', [])
        if isinstance(source, str):
            source = [source]
        
        # Check if this is a YAML front matter cell
        if len(source) == 0 or not source[0].strip().startswith('---'):
            return False
        
        # Ensure first cell is raw type (needed for Quarto)
        first_cell['cell_type'] = 'raw'
        first_cell['metadata'] = {}
        
        # Parse the YAML to extract title/author/date
        yaml_data = parse_yaml_frontmatter(source)
        
        # Create formatted header cell
        header_cell = create_formatted_header_cell(yaml_data)
        
        if header_cell:
            # Check if second cell already exists
            if len(notebook['cells']) > 1:
                second_cell = notebook['cells'][1]
                # If it's a markdown cell with the title, replace it
                if second_cell.get('cell_type') == 'markdown':
                    second_source = ''.join(second_cell.get('source', []))
                    if yaml_data and 'title' in yaml_data:
                        if yaml_data['title'] in second_source:
                            # Replace existing header
                            notebook['cells'][1] = header_cell
                        else:
                            # Insert new header before existing content
                            notebook['cells'].insert(1, header_cell)
                else:
                    # Insert after raw cell
                    notebook['cells'].insert(1, header_cell)
            else:
                # No second cell exists, insert new one
                notebook['cells'].insert(1, header_cell)
        
        # Write back the notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Fixed: {notebook_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {notebook_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Find and fix all .ipynb files"""
    notebook_files = glob.glob('**/*.ipynb', recursive=True)
    
    if not notebook_files:
        print("No .ipynb files found")
        return
    
    print(f"Found {len(notebook_files)} notebook(s) to process\n")
    
    fixed_count = 0
    for notebook_path in notebook_files:
        if fix_notebook_display(notebook_path):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} out of {len(notebook_files)} notebooks")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
