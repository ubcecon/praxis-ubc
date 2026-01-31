#!/usr/bin/env python3
"""
Re-render Self-Contained QMD Files

This script automates the process of re-rendering .qmd files that need to be
self-contained HTML files. These files are excluded from the Docker build
because they require packages/datasets not available in the container.

The script:
1. Parses _quarto.yml to identify excluded .qmd files (prefixed with !)
2. Verifies each file has correct self-contained headers
3. Renders each file using `quarto render`
4. Copies the generated HTML from _site to the source directory

Usage:
    python re_render_qmd_files.py [--dry-run] [--file FILENAME]

Options:
    --dry-run       Show what would be done without actually rendering (python re_render_qmd_files.py --dry-run)
    --file FILENAME Only render a specific file (partial match supported)
    --skip-verify   Skip header verification
    --verbose       Show detailed output

Date: 2025-01-24
Updated: 2026-01-30 - Fixed: explicitly pass cwd to subprocess.run() for reliable
                      rendering from project directory (required for _quarto.yml)
"""

import argparse
import os
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
    """Result of a render operation."""
    file_path: Path
    success: bool
    error_message: Optional[str] = None
    html_copied: bool = False


class QMDReRenderer:
    """Handles re-rendering of self-contained QMD files."""

    # Required header patterns for self-contained files
    REQUIRED_EXECUTE_PATTERNS = {
        'eval': r'eval:\s*true',
        'echo': r'echo:\s*true',
        'output': r'output:\s*true',
    }

    REQUIRED_HTML_PATTERNS = {
        'embed-resources': r'embed-resources:\s*true',
        'self-contained-math': r'self-contained-math:\s*true',
    }

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.project_root = repo_root / 'project'
        self.quarto_yml = self.project_root / '_quarto.yml'
        self.verbose = verbose

    def log(self, message: str, force: bool = False):
        """Print message if verbose mode or forced."""
        if self.verbose or force:
            print(message)

    def parse_excluded_files(self) -> list[str]:
        """Parse _quarto.yml to find files excluded from render (prefixed with !)."""
        if not self.quarto_yml.exists():
            raise FileNotFoundError(f"_quarto.yml not found at {self.quarto_yml}")

        excluded_files = []
        content = self.quarto_yml.read_text(encoding='utf-8')

        # Find lines with ! prefix in the render section
        # Pattern matches: - "!docs/path/file.qmd" or - "!in_progress/*"
        pattern = r'-\s*["\']?!([^"\'#\n]+\.qmd)["\']?'
        matches = re.findall(pattern, content)

        for match in matches:
            # Clean up the path
            file_path = match.strip()
            # Skip wildcard patterns
            if '*' not in file_path:
                excluded_files.append(file_path)

        return excluded_files

    def verify_headers(self, qmd_path: Path) -> tuple[bool, list[str]]:
        """Verify that a QMD file has the required self-contained headers."""
        if not qmd_path.exists():
            return False, [f"File not found: {qmd_path}"]

        content = qmd_path.read_text(encoding='utf-8')

        # Extract YAML front matter
        yaml_match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
        if not yaml_match:
            return False, ["No YAML front matter found"]

        yaml_content = yaml_match.group(1)
        missing = []

        # Check execute section
        for name, pattern in self.REQUIRED_EXECUTE_PATTERNS.items():
            if not re.search(pattern, yaml_content, re.IGNORECASE):
                missing.append(f"execute.{name}: true")

        # Check html section (under format)
        for name, pattern in self.REQUIRED_HTML_PATTERNS.items():
            if not re.search(pattern, yaml_content, re.IGNORECASE):
                missing.append(f"format.html.{name}: true")

        return len(missing) == 0, missing

    def _temporarily_unexclude_file(self, rel_path: str) -> str:
        """Temporarily remove exclusion for a file from _quarto.yml.

        Returns the original line (with newline) for restoration.
        """
        content = self.quarto_yml.read_text(encoding='utf-8')

        # Find and comment out the exclusion line
        # Pattern: - "!docs/path/file.qmd" or - '!docs/path/file.qmd'
        pattern = rf'([ \t]*-[ \t]*["\']?!{re.escape(rel_path)}["\']?[ \t]*\r?\n)'

        match = re.search(pattern, content)
        if match:
            original_line = match.group(1)
            # Comment out the line (preserving line ending)
            commented = f"  # TEMP_DISABLED:{original_line.rstrip()}\n"
            new_content = content.replace(original_line, commented)
            self.quarto_yml.write_text(new_content, encoding='utf-8')
            return original_line
        return ""

    def _restore_exclusion(self, original_line: str):
        """Restore the original exclusion line in _quarto.yml."""
        if not original_line:
            return

        content = self.quarto_yml.read_text(encoding='utf-8')
        # Find the commented line and restore it
        commented_pattern = rf'[ \t]*# TEMP_DISABLED:{re.escape(original_line.rstrip())}\r?\n'
        new_content = re.sub(commented_pattern, original_line, content)
        self.quarto_yml.write_text(new_content, encoding='utf-8')

    def _get_directory_contents(self, directory: Path) -> set[str]:
        """Get set of all files and directories in a directory."""
        contents = set()
        if directory.exists():
            for item in directory.iterdir():
                contents.add(item.name)
        return contents

    def _cleanup_render_artifacts(self, qmd_path: Path, pre_render_contents: set[str], target_html: str):
        """Remove any new files/directories created during render, except target HTML."""
        parent_dir = qmd_path.parent
        current_contents = self._get_directory_contents(parent_dir)

        # Find new items (created during render)
        new_items = current_contents - pre_render_contents

        # Also always clean up these patterns even if they existed before
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
        """Render a single QMD file and copy the output HTML."""
        result = RenderResult(file_path=qmd_path, success=False)

        if not qmd_path.exists():
            result.error_message = f"File not found: {qmd_path}"
            return result

        # Determine paths
        relative_path = qmd_path.relative_to(self.project_root)
        html_filename = qmd_path.stem + '.html'

        # Output path in _site
        site_html_path = self.project_root / '_site' / relative_path.parent / html_filename

        # Destination path (same directory as source)
        dest_html_path = qmd_path.parent / html_filename

        if dry_run:
            print(f"  [DRY RUN] Would render: {qmd_path}")
            print(f"  [DRY RUN] Output will be at: {dest_html_path}")
            print(f"  [DRY RUN] (excluded files render directly to source dir)")
            result.success = True
            return result

        # Record directory contents before render for cleanup
        pre_render_contents = self._get_directory_contents(qmd_path.parent)

        # Temporarily remove exclusion so file renders with project settings
        original_exclusion = None
        try:
            # Use forward slashes for quarto command (works on all platforms)
            relative_path_str = relative_path.as_posix()

            # Temporarily unexclude the file from _quarto.yml
            # This ensures it renders WITH navbar and theme from project
            self.log(f"  Temporarily enabling file in _quarto.yml...")
            original_exclusion = self._temporarily_unexclude_file(relative_path_str)

            # Record time before render to detect newly created files
            render_start_time = time.time()

            # Run quarto render from the project directory
            self.log(f"  Running: quarto render {relative_path_str}")
            self.log(f"  Working directory: {self.project_root}")

            process = subprocess.run(
                ['quarto', 'render', relative_path_str],
                capture_output=True,
                text=True,
                cwd=self.project_root,  # Explicitly set working directory
                # No timeout - some notebooks take longer to render
            )

            if process.returncode != 0:
                result.error_message = f"Quarto render failed:\n{process.stderr}"
                if process.stdout:
                    result.error_message += f"\nstdout:\n{process.stdout}"
                return result

            # Check if HTML was freshly created in _site (project render behavior)
            site_is_fresh = (site_html_path.exists() and
                           site_html_path.stat().st_mtime >= render_start_time)

            # Check if HTML was freshly created in source directory (single file render behavior)
            # When rendering a single file with `quarto render file.qmd`, Quarto outputs
            # to the source directory, not _site, regardless of project configuration.
            dest_is_fresh = (dest_html_path.exists() and
                            dest_html_path.stat().st_mtime >= render_start_time)

            if site_is_fresh:
                # HTML was created in _site/, copy to source directory
                self.log(f"  Copying: {site_html_path} -> {dest_html_path}")
                shutil.copy2(site_html_path, dest_html_path)
                result.success = True
                result.html_copied = True
            elif dest_is_fresh:
                # HTML was created directly in source directory (single file render)
                # No copy needed - file is already in the right place
                self.log(f"  HTML created directly in source directory: {dest_html_path}")
                result.success = True
                result.html_copied = False  # No copy was needed
            else:
                result.error_message = f"HTML not freshly created at expected locations:\n"
                result.error_message += f"  - {dest_html_path}\n"
                result.error_message += f"  - {site_html_path}"
                return result

            # Clean up unwanted files created by Quarto
            self._cleanup_render_artifacts(qmd_path, pre_render_contents, html_filename)

        except Exception as e:
            result.error_message = f"Unexpected error: {str(e)}"
        finally:
            # Always restore the exclusion in _quarto.yml
            if original_exclusion:
                self.log(f"  Restoring exclusion in _quarto.yml...")
                self._restore_exclusion(original_exclusion)

        return result

    def render_all(self, dry_run: bool = False, file_filter: Optional[str] = None,
                   skip_verify: bool = False) -> tuple[list[RenderResult], list[RenderResult]]:
        """Render all excluded QMD files."""
        excluded_files = self.parse_excluded_files()

        if not excluded_files:
            print("No excluded QMD files found in _quarto.yml")
            return [], []

        print(f"Found {len(excluded_files)} files to render:")
        for f in excluded_files:
            print(f"  - {f}")
        print()

        # Filter if specified
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

            # Verify headers
            if not skip_verify:
                is_valid, missing = self.verify_headers(qmd_path)
                if not is_valid:
                    print(f"  WARNING: Missing required headers:")
                    for m in missing:
                        print(f"    - {m}")
                    if not dry_run:
                        response = input("  Continue anyway? [y/N]: ").strip().lower()
                        if response != 'y':
                            result = RenderResult(
                                file_path=qmd_path,
                                success=False,
                                error_message="Skipped due to missing headers"
                            )
                            failed.append(result)
                            continue

            # Render
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
    """Print a summary of the render results."""
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
            print(f"  [FAIL] {r.file_path.name}")
            if r.error_message:
                # Truncate long error messages
                error_lines = r.error_message.split('\n')
                for line in error_lines[:5]:
                    print(f"         {line}")
                if len(error_lines) > 5:
                    print(f"         ... ({len(error_lines) - 5} more lines)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Re-render self-contained QMD files for prAxIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python re_render_qmd_files.py                  # Render all excluded files
  python re_render_qmd_files.py --dry-run        # Show what would be done
  python re_render_qmd_files.py --file soci_415  # Only render files matching 'soci_415'
  python re_render_qmd_files.py --verbose        # Show detailed output
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without actually rendering')
    parser.add_argument('--file', type=str, metavar='FILTER',
                        help='Only render files matching this filter')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip header verification')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output')

    args = parser.parse_args()

    # Determine repository root
    # Script is at: praxis-ubc/meta/scripts/re_render_qmd_files.py
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent

    # Verify we're in the right place
    if not (repo_root / 'project' / '_quarto.yml').exists():
        print(f"Error: Cannot find project/_quarto.yml from {repo_root}")
        print("Please run this script from the praxis-ubc repository.")
        sys.exit(1)

    print("=" * 60)
    print("prAxIs QMD Re-Renderer")
    print("=" * 60)
    print(f"Repository root: {repo_root}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Check for quarto
    try:
        result = subprocess.run(['quarto', '--version'], capture_output=True, text=True)
        print(f"Quarto version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Quarto is not installed or not in PATH")
        print("Please install Quarto from https://quarto.org/")
        sys.exit(1)
    print()

    # Create renderer and run
    renderer = QMDReRenderer(repo_root, verbose=args.verbose)

    try:
        successful, failed = renderer.render_all(
            dry_run=args.dry_run,
            file_filter=args.file,
            skip_verify=args.skip_verify
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print_summary(successful, failed)

    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
