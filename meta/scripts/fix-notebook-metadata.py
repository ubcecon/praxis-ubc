import json
import sys
import os
import glob

def fix_notebook_metadata(notebook_path):
    """Fix the first cell containing YAML front matter to be a Raw cell"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        if not notebook.get('cells') or len(notebook['cells']) == 0:
            return False
        
        first_cell = notebook['cells'][0]
        
        # Check if first cell contains YAML front matter
        if (first_cell.get('cell_type') == 'markdown' and 
            isinstance(first_cell.get('source'), list) and 
            len(first_cell['source']) > 0 and 
            first_cell['source'][0].startswith('---')):
            
            # Convert to raw cell
            first_cell['cell_type'] = 'raw'
            first_cell['metadata'] = {}
            
            # Write back the fixed notebook
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            
            print(f"Fixed metadata cell in: {notebook_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {notebook_path}: {e}")
        return False

def main():
    """Find and fix all .ipynb files in the current directory and subdirectories"""
    notebook_files = glob.glob('**/*.ipynb', recursive=True)
    
    if not notebook_files:
        print("No .ipynb files found")
        return
    
    fixed_count = 0
    for notebook_path in notebook_files:
        if fix_notebook_metadata(notebook_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} out of {len(notebook_files)} notebooks")

if __name__ == "__main__":
    main()
