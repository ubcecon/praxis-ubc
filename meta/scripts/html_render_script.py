#As of 2025-11-23 has not been tested yet, will test with new notebooks or if needs to be used in COMET
"""
Quarto Notebook Rendering Automation Script

Automates the creation of .html files for prAxIs and changed to the .dockerfile and _quarto.yml
"""
import os
import sys
import shutil
import subprocess
import re
from pathlib import Path
from typing import Tuple

#CONFIGURE PATHS AND NOTEBOOKS
REPO_BASE_PATH = r"C:\Users\alexr\OneDrive\Documents\GitHub\praxis-ubc" #Change here

NOTEBOOKS = [
    "intro_to_deep_learning/intro_to_fundamental_ML.qmd", #Example on right now
    # Add more notebooks here:
]

# CONFIGURATION STRINGS (IN CASE MORE STUFF HAS TO BE ADDED .QMD HEADERS)
EXECUTE_CONFIG = """execute:
  eval: true
  echo: true
  output: true"""

HTML_CONFIG = """  html:
    embed-resources: true
    self-contained-math: true"""

# FUNCTIONS
def validate_paths(repo_path: str) -> bool:
    """Validate that the repository path exists and has expected structure."""
    repo = Path(repo_path)
    
    if not repo.exists():
        print(f"ERROR: Repository path does not exist: {repo_path}")
        return False
    
    required_paths = [
        (repo / "project" / "docs", "project/docs"),
        (repo / "project" / "_site", "project/_site"),
        (repo / "meta" / "building" / ".dockerfile", ".dockerfile"),
        (repo / "project" / "_quarto.yml", "_quarto.yml")
    ]
    
    for path, name in required_paths:
        if not path.exists():
            print(f"ERROR: Required path not found: {name}")
            return False
    
    print("All required paths validated")
    return True

def parse_notebook_path(notebook_relative_path: str) -> Tuple[str, str, str]:
    """
    Parse notebook path into components.
    
    Returns: (directory_path, filename_without_ext, full_filename)
    Example: ("intro_to_deep_learning", "intro_to_fundamental_ML", "intro_to_fundamental_ML.qmd")
    """
    path = Path(notebook_relative_path)
    directory = path.parent.as_posix()
    filename_without_ext = path.stem
    full_filename = path.name
    
    return directory, filename_without_ext, full_filename


def modify_qmd_header(qmd_path: Path) -> bool:
    """Add execute and html configuration to QMD file header."""
    print(f"\nModifying QMD header: {qmd_path.name}")
    
    if not qmd_path.exists():
        print(f"ERROR: QMD file not found: {qmd_path}")
        return False
    
    with open(qmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find YAML header
    yaml_pattern = r'^---\n(.*?\n)---'
    match = re.search(yaml_pattern, content, re.DOTALL | re.MULTILINE)
    
    if not match:
        print("ERROR: Could not find YAML header in QMD file")
        return False
    
    yaml_content = match.group(1)
    
    # Check if already configured
    if 'execute:' in yaml_content and 'eval: true' in yaml_content:
        print("Execute configuration already exists, skipping")
        return True
    
    # Add configurations
    if 'format:' in yaml_content:
        new_yaml = yaml_content.replace('format:', f'{EXECUTE_CONFIG}\nformat:')
        
        if 'html:' in new_yaml:
            new_yaml = re.sub(
                r'(html:\n(?:    .*\n)*)',
                lambda m: m.group(1).rstrip() + f'\n{HTML_CONFIG}\n',
                new_yaml
            )
        else:
            new_yaml = re.sub(r'(format:\n)', f'\\1{HTML_CONFIG}\n', new_yaml)
    else:
        new_yaml = yaml_content.rstrip() + f'\n{EXECUTE_CONFIG}\nformat:\n{HTML_CONFIG}\n'
    
    new_content = content.replace(yaml_content, new_yaml)
    
    with open(qmd_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Header modified successfully")
    return True

def render_notebook(qmd_path: Path) -> bool:
    """Render the Quarto notebook."""
    print(f"\nRendering notebook: {qmd_path.name}")
    
    original_dir = os.getcwd()
    os.chdir(qmd_path.parent)
    
    try:
        result = subprocess.run(
            ['quarto', 'render', qmd_path.name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout can be changed here if needed
        )
        
        if result.returncode != 0:
            print(f"ERROR: Quarto render failed")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("Notebook rendered successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("ERROR: Quarto render timed out after 10 minutes")
        return False
    except FileNotFoundError:
        print("ERROR: 'quarto' command not found. Is Quarto installed?")
        return False
    finally:
        os.chdir(original_dir)


def copy_rendered_html(repo_path: Path, notebook_relative_path: str) -> bool:
    """Copy rendered HTML from _site back to source directory."""
    print(f"\nCopying rendered HTML file")
    
    directory, filename_without_ext, _ = parse_notebook_path(notebook_relative_path)
    html_filename = f"{filename_without_ext}.html"
    
    source_html = repo_path / "project" / "_site" / "docs" / directory / html_filename
    dest_html = repo_path / "project" / "docs" / directory / html_filename
    
    if not source_html.exists():
        print(f"ERROR: Rendered HTML not found at: {source_html}")
        return False
    
    shutil.copy2(source_html, dest_html)
    
    if dest_html.exists():
        print(f"HTML copied to: {dest_html}")
        return True
    else:
        print("ERROR: Failed to copy HTML file")
        return False


def update_dockerfile(repo_path: Path, notebook_relative_path: str) -> bool:
    """Update .dockerfile to remove .qmd and copy .html."""
    print(f"\nUpdating .dockerfile")
    
    dockerfile_path = repo_path / "meta" / "building" / ".dockerfile"
    directory, filename_without_ext, full_filename = parse_notebook_path(notebook_relative_path)
    html_filename = f"{filename_without_ext}.html"
    
    rm_line = f"RUN rm -f ./docs/{directory}/{full_filename}"
    copy_line = f"COPY ./project/docs/{directory}/{html_filename} /app/output/docs/{directory}/"
    
    with open(dockerfile_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find insertion points
    rm_section_idx = None
    copy_section_idx = None
    
    for i, line in enumerate(lines):
        if '#Removes the rendered.qmd' in line or '# Removes the rendered.qmd' in line:
            rm_section_idx = i
        if '# Copy pre-rendered HTML file' in line:
            copy_section_idx = i
    
    # Check if already exists
    dockerfile_content = ''.join(lines)
    
    if rm_line not in dockerfile_content and rm_section_idx is not None:
        lines.insert(rm_section_idx + 1, rm_line + "\n")
        copy_section_idx += 1
        print(f"Added rm line: {rm_line}")
    
    if copy_line not in dockerfile_content and copy_section_idx is not None:
        lines.insert(copy_section_idx + 1, copy_line + "\n")
        print(f"Added copy line: {copy_line}")
    
    with open(dockerfile_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(".dockerfile updated")
    return True

def update_quarto_yml(repo_path: Path, notebook_relative_path: str) -> bool:
    """Update _quarto.yml to ignore the .qmd file."""
    print(f"\nUpdating _quarto.yml")
    
    quarto_yml_path = repo_path / "project" / "_quarto.yml"
    directory, _, full_filename = parse_notebook_path(notebook_relative_path)
    
    ignore_line = f'  - "!docs/{directory}/{full_filename}"'
    
    with open(quarto_yml_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if already exists
    if ignore_line.strip() in ''.join(lines):
        print("Ignore line already exists")
        return True
    
    # Find last ignore line
    last_ignore_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('- "!'):
            last_ignore_idx = i
    
    if last_ignore_idx is not None:
        lines.insert(last_ignore_idx + 1, ignore_line + "\n")
        print(f"Added ignore line: {ignore_line}")
    else:
        print("WARNING: Could not find ignore section in _quarto.yml")
        return False
    
    with open(quarto_yml_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("_quarto.yml updated")
    return True


def process_notebook(repo_path: Path, notebook_relative_path: str) -> bool:
    """Process a single notebook through the entire workflow."""
    print(f"Processing: {notebook_relative_path}")
    
    directory, filename_without_ext, full_filename = parse_notebook_path(notebook_relative_path)
    qmd_path = repo_path / "project" / "docs" / directory / full_filename
    
    steps = [
        ("Modify QMD header", lambda: modify_qmd_header(qmd_path)),
        ("Render notebook", lambda: render_notebook(qmd_path)),
        ("Copy HTML", lambda: copy_rendered_html(repo_path, notebook_relative_path)),
        ("Update Dockerfile", lambda: update_dockerfile(repo_path, notebook_relative_path)),
        ("Update _quarto.yml", lambda: update_quarto_yml(repo_path, notebook_relative_path)),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\nFailed at step: {step_name}")
            return False
    
    print("\n" + "="*80)
    print(f"Successfully processed: {notebook_relative_path}")
    print("="*80)
    return True

# main function 
def main():
    print("Quarto Notebook Rendering Automation")
    print(f"Repository: {REPO_BASE_PATH}")
    print(f"Notebooks to process: {len(NOTEBOOKS)}\n")
    
    repo_path = Path(REPO_BASE_PATH)
    
    if not validate_paths(REPO_BASE_PATH):
        print("\nPath validation failed")
        sys.exit(1)
    
    success_count = 0
    failed_notebooks = []
    
    for notebook in NOTEBOOKS:
        try:
            if process_notebook(repo_path, notebook):
                success_count += 1
            else:
                failed_notebooks.append(notebook)
        except Exception as e:
            print(f"\nERROR processing {notebook}: {str(e)}")
            failed_notebooks.append(notebook)
    
    # Summary print out
    print(f"Total: {len(NOTEBOOKS)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_notebooks)}")
    
    if failed_notebooks:
        print("\nFailed notebooks:")
        for nb in failed_notebooks:
            print(f"  - {nb}")
        sys.exit(1)
    else:
        print("\nAll notebooks processed successfully")
    
    sys.exit(0)


if __name__ == "__main__":
    main()
