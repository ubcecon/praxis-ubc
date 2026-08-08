# Builds pvz_meta.json for the path visualization page: scans the rendered docs/ HTML for <meta> description/image that can be used on the page
import json
import os
import posixpath
import re
import sys
from html.parser import HTMLParser

KEYS = ("description", "og:description", "og:image", "twitter:image")
FRONT_MATTER_IMAGE = re.compile(r"^image:\s*(.+?)\s*$")


class MetaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.metas = {}
        self.done = False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            key = a.get("name") or a.get("property")
            if key in KEYS and key not in self.metas:
                self.metas[key] = a.get("content")
        elif tag == "body":
            self.done = True


def page_meta(path):
    parser = MetaParser()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        while not parser.done:
            chunk = fh.read(65536)
            if not chunk:
                break
            parser.feed(chunk)
    # Same preference order as the page's per-hover fallback fetch
    desc = parser.metas.get("description") or parser.metas.get("og:description")
    img = parser.metas.get("og:image") or parser.metas.get("twitter:image")
    return desc, img


def source_image(src_root, rel):
    # Quarto only writes og:image when website.open-graph is on, and this site so there's no thumbnail in the rendered HTML to scrape. We take it directly from notebook front matter. 
    qmd = os.path.join(src_root, rel[: -len(".html")] + ".qmd")
    if not os.path.isfile(qmd):
        return None
    with open(qmd, encoding="utf-8", errors="ignore") as fh:
        if fh.readline().strip() != "---":
            return None
        for line in fh:
            if line.strip() == "---":
                break
            m = FRONT_MATTER_IMAGE.match(line)
            if m:
                return m.group(1).strip("\"'")
    return None


def resolve_image(img, rel):
    # Front matter writes image in three different ways, so flatten them all to a standardized site-root-relative path. 
    if img.startswith(("http://", "https://")):
        return img
    if img.startswith("/"):
        return img.lstrip("/")  # project-root absolute, ie /media/comet_placeholder.png
    return posixpath.normpath(posixpath.join(posixpath.dirname(rel), img))


def main():
    root, out_path = sys.argv[1], sys.argv[2]
    src_root = sys.argv[3] if len(sys.argv) > 3 else root
    index = {}
    for dirpath, _, files in os.walk(os.path.join(root, "docs")):
        for name in files:
            if not name.endswith(".html"):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                desc, img = page_meta(full)
            except Exception as e:
                print(f"pvz meta index: skipped {rel} ({e})")
                continue
            if not img:
                raw = source_image(src_root, rel)
                if raw:
                    img = resolve_image(raw, rel)
            if desc or img:
                index[rel] = {"description": desc, "image": img}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    print(f"pvz meta index: {len(index)} pages -> {out_path}")


if __name__ == "__main__":
    main()
