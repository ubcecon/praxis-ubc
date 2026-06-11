#!/usr/bin/env python3

# When PyStata starts up, Stata prints its license banner, which includes the
# serial number and the name the license is registered to, for example:
#
#     Stata license: Unlimited-user network, expiring 19 Aug 2026
#     Serial number: 401909301439
#       Licensed to: Your Name
#                    UBC
#
# Left as-is, that banner ends up baked into the published HTML, putting a real
# person's name and license serial on a public website. This script adds
# "#| output: false" to the stata_setup.config() cell so the banner stays hidden
# (the cell still runs, Stata still starts). Re-render afterwards to apply it.
#
# Usage:
#   python meta/scripts/suppress_stata_license_banner.py            # patch the notebooks
#   python meta/scripts/suppress_stata_license_banner.py --dry-run  # just show what would change

import argparse
import re
from pathlib import Path

NOTEBOOK_DIR = "project/docs/5_Research/econ490-pystata"

# The ```{python} fence, any options already on the cell, then the config call.
CONFIG_CELL = re.compile(
    r"(```\{python\}\r?\n)((?:#\|[^\n]*\r?\n)*)(import stata_setup\r?\nstata_setup\.config\()"
)


def patch(qmd, dry_run):
    text = qmd.read_text(encoding="utf-8")
    m = CONFIG_CELL.search(text)
    if not m or "output: false" in m.group(2):
        return False
    if not dry_run:
        new = text[:m.start()] + m.group(1) + "#| output: false\n" + m.group(2) + m.group(3) + text[m.end():]
        qmd.write_text(new, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Hide the Stata license banner from the PyStata notebooks.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing anything")
    args = parser.parse_args()

    notebooks = sorted((Path(__file__).resolve().parents[2] / NOTEBOOK_DIR).glob("*.qmd"))
    patched = [q.name for q in notebooks if patch(q, args.dry_run)]

    for name in patched:
        print(("would patch " if args.dry_run else "patched ") + name)
    print(f"\n{len(patched)} notebook(s) {'to patch' if args.dry_run else 'patched'}.")
    if patched and not args.dry_run:
        print("Re-render with re_render_qmd_files.py for it to take effect.")


if __name__ == "__main__":
    main()
