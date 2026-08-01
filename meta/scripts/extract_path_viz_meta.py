# Builds pvz_meta.json for the path visualization page: scans the rendered docs/ HTML for <meta> description/image that can be used on the page
import json
import os
import sys
from html.parser import HTMLParser

KEYS = ("description", "og:description", "og:image", "twitter:image")


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


def main():
    root, out_path = sys.argv[1], sys.argv[2]
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
            if desc or img:
                index[rel] = {"description": desc, "image": img}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False)
    print(f"pvz meta index: {len(index)} pages -> {out_path}")


if __name__ == "__main__":
    main()
