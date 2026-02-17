#!/usr/bin/env python3

# Re-render self-contained QMD files excluded from the Docker build.
# See meta/scripts/Using_the_re-rended_script.md for full documentation.


import argparse
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RenderResult:
    file_path: Path
    success: bool
    error_message: Optional[str] = None


class QMDReRenderer:

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.project_root = repo_root / 'project'
        self.quarto_yml = self.project_root / '_quarto.yml'
        self.verbose = verbose

    def log(self, message: str, force: bool = False):
        if self.verbose or force:
            print(message)

    # Find files excluded from render in _quarto.yml (the ones with !prefix)
    def parse_excluded_files(self) -> list[str]:
        if not self.quarto_yml.exists():
            raise FileNotFoundError(f"_quarto.yml not found at {self.quarto_yml}")

        excluded_files = []
        content = self.quarto_yml.read_text(encoding='utf-8')

        # Matches: - "!docs/path/file.qmd" or - '!docs/path/file.qmd'
        pattern = r'-\s*["\']?!([^"\'#\n]+\.qmd)["\']?'
        matches = re.findall(pattern, content)

        for match in matches:
            file_path = match.strip()
            if '*' not in file_path:
                excluded_files.append(file_path)

        return excluded_files

    # Verify the QMD has required self-contained headers
    REQUIRED_HEADERS = {
        'eval': r'eval:\s*true',
        'echo': r'echo:\s*true',
        'output': r'output:\s*true',
        'embed-resources': r'embed-resources:\s*true',
        'self-contained-math': r'self-contained-math:\s*true',
    }

    def verify_headers(self, qmd_path: Path) -> list[str]:
        content = qmd_path.read_text(encoding='utf-8')
        yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not yaml_match:
            return ["No YAML front matter found"]

        yaml_content = yaml_match.group(1)
        return [name for name, pattern in self.REQUIRED_HEADERS.items()
                if not re.search(pattern, yaml_content, re.IGNORECASE)]

    # Comment out the exclusion line so the file renders with project settings (ie the Praxis Theme, Navbar etc)
    def _temporarily_unexclude_file(self, rel_path: str) -> str:
        content = self.quarto_yml.read_text(encoding='utf-8')
        pattern = rf'([ \t]*-[ \t]*["\']?!{re.escape(rel_path)}["\']?[^\r\n]*\r?\n)'

        match = re.search(pattern, content)
        if match:
            original_line = match.group(1)
            commented = f"  # TEMP_DISABLED:{original_line.rstrip()}\n"
            new_content = content.replace(original_line, commented)
            self.quarto_yml.write_text(new_content, encoding='utf-8')
            return original_line
        return ""

    # Put the exclusion line back in _quarto.yml
    def _restore_exclusion(self, original_line: str):
        if not original_line:
            return

        content = self.quarto_yml.read_text(encoding='utf-8')
        commented_pattern = rf'[ \t]*# TEMP_DISABLED:{re.escape(original_line.rstrip())}\r?\n'
        new_content = re.sub(commented_pattern, original_line, content)
        self.quarto_yml.write_text(new_content, encoding='utf-8')

    def _get_directory_contents(self, directory: Path) -> set[str]:
        contents = set()
        if directory.exists():
            for item in directory.iterdir():
                contents.add(item.name)
        return contents

    # Remove intermediate files Quarto creates during render
    def _cleanup_render_artifacts(self, qmd_path: Path, pre_render_contents: set[str], target_html: str):
        parent_dir = qmd_path.parent
        current_contents = self._get_directory_contents(parent_dir)
        new_items = current_contents - pre_render_contents

        # Always clean these up even if they existed before
        base_name = qmd_path.stem
        always_clean = {
            f"{base_name}.ipynb",
            f"{base_name}.quarto_ipynb",
            f"{base_name}_files",
            "lib",  # Quarto lib directory
        }

        items_to_clean = (new_items | always_clean) - {target_html}

        for item_name in items_to_clean:
            item_path = parent_dir / item_name
            if item_path.exists():
                try:
                    if item_path.is_dir():
                        shutil.rmtree(item_path)
                        self.log(f"  Cleaned up directory: {item_name}")
                    else:
                        item_path.unlink()
                        self.log(f"  Cleaned up file: {item_name}")
                except Exception as e:
                    self.log(f"  Warning: Could not remove {item_name}: {e}", force=True)

    def render_file(self, qmd_path: Path, dry_run: bool = False) -> RenderResult:
        result = RenderResult(file_path=qmd_path, success=False)
        relative_path = qmd_path.relative_to(self.project_root)
        html_filename = qmd_path.stem + '.html'
        site_html_path = self.project_root / '_site' / relative_path.parent / html_filename
        dest_html_path = qmd_path.parent / html_filename

        if dry_run:
            print(f"  [DRY RUN] Would render: {qmd_path}")
            print(f"  [DRY RUN] Output: {dest_html_path}")
            result.success = True
            return result

        pre_render_contents = self._get_directory_contents(qmd_path.parent)
        original_exclusion = None

        try:
            # Forward slashes work on all platforms for quarto
            relative_path_str = relative_path.as_posix()

            self.log(f"  Temporarily enabling file in _quarto.yml...")
            original_exclusion = self._temporarily_unexclude_file(relative_path_str)
            render_start_time = time.time()

            self.log(f"  Running: quarto render {relative_path_str}")
            process = subprocess.run(
                ['quarto', 'render', relative_path_str],
                text=True,
                cwd=self.project_root,
            )

            if process.returncode != 0:
                result.error_message = "Quarto render failed (see output above)"
                return result

            # Quarto may output to _site/ (project render) or source dir (single file render)
            site_is_fresh = (site_html_path.exists() and
                           site_html_path.stat().st_mtime >= render_start_time)
            dest_is_fresh = (dest_html_path.exists() and
                            dest_html_path.stat().st_mtime >= render_start_time)

            if site_is_fresh:
                self.log(f"  Copying: {site_html_path} -> {dest_html_path}")
                shutil.copy2(site_html_path, dest_html_path)
                result.success = True
            elif dest_is_fresh:
                self.log(f"  HTML created directly in source directory: {dest_html_path}")
                result.success = True
            else:
                result.error_message = f"HTML not found after render"
                return result

            self._cleanup_render_artifacts(qmd_path, pre_render_contents, html_filename)

        finally:
            if original_exclusion:
                self.log(f"  Restoring exclusion in _quarto.yml...")
                self._restore_exclusion(original_exclusion)

        return result

    def render_all(self, dry_run: bool = False, file_filter: Optional[str] = None) -> tuple[list[RenderResult], list[RenderResult]]:
        excluded_files = self.parse_excluded_files()

        if not excluded_files:
            print("No excluded QMD files found in _quarto.yml")
            return [], []

        print(f"Found {len(excluded_files)} files to render:")
        for f in excluded_files:
            print(f"  - {f}")
        print()

        if file_filter:
            excluded_files = [f for f in excluded_files if file_filter.lower() in f.lower()]
            if not excluded_files:
                print(f"No files matching filter: {file_filter}")
                return [], []
            print(f"Filtering to {len(excluded_files)} files matching '{file_filter}'")
            print()

        successful = []
        failed = []

        for i, rel_path in enumerate(excluded_files, 1):
            qmd_path = self.project_root / rel_path
            print(f"[{i}/{len(excluded_files)}] Processing: {rel_path}")

            missing = self.verify_headers(qmd_path)
            if missing:
                result = RenderResult(
                    file_path=qmd_path, success=False,
                    error_message=f"Missing headers: {', '.join(missing)}"
                )
                failed.append(result)
                print(f"  FAILED: {result.error_message}")
                print()
                continue

            result = self.render_file(qmd_path, dry_run=dry_run)

            if result.success:
                print(f"  SUCCESS")
                successful.append(result)
            else:
                print(f"  FAILED: {result.error_message}")
                failed.append(result)

            print()

        return successful, failed


def print_summary(successful: list[RenderResult], failed: list[RenderResult]):
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(successful) + len(failed)
    print(f"Total files: {total}")
    print(f"Successful:  {len(successful)}")
    print(f"Failed:      {len(failed)}")
    print()

    if successful:
        print("Successfully rendered:")
        for r in successful:
            print(f"  [OK] {r.file_path.name}")
        print()

    if failed:
        print("Failed to render:")
        for r in failed:
            print(f"  [FAIL] {r.file_path.name}: {r.error_message or 'unknown error'}")
        print()


#Check asssociated .md file for more info on parser arguments
def main():
    parser = argparse.ArgumentParser(
        description='Re-render self-contained QMD files for Praxis')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually rendering')
    parser.add_argument('--file', type=str, metavar='FILTER',
                        help='Only render files matching this filter')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent

    print("prAxIs QMD Re-Renderer")
    print(f"Repository root: {repo_root}")
    print(f"Dry run: {args.dry_run}")
    print()

    try:
        result = subprocess.run(['quarto', '--version'], capture_output=True, text=True)
        print(f"Quarto version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Quarto is not installed or not in PATH")
        print("Please install Quarto from https://quarto.org/")
        sys.exit(1)
    print()

    renderer = QMDReRenderer(repo_root, verbose=args.verbose)

    try:
        successful, failed = renderer.render_all(
            dry_run=args.dry_run,
            file_filter=args.file,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

    print_summary(successful, failed)

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
