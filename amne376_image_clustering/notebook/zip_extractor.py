import zipfile
from pathlib import Path

class ZipExtractor:
    def __init__(self, source_zip, output_dir):
        self.source_zip = Path(source_zip)
        if not self.source_zip.is_file():
            raise FileNotFoundError(f"ZIP file not found: {self.source_zip}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, strip_root_folder: bool = True):
        """
        Extracts the ZIP file into a subfolder named after the ZIP stem.
        If strip_root_folder is True and the archive has a single top-level
        folder, that folder name is removed in the output.
        """
        stem = self.source_zip.stem
        target_dir = self.output_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.source_zip, 'r') as zip_ref:
            names = zip_ref.namelist()

            # Detect single root folder
            if strip_root_folder:
                # Get the prefix before first slash for each name
                prefixes = {n.split('/', 1)[0] for n in names if '/' in n}
                if len(prefixes) == 1 and prefixes.pop() == stem:
                    # All paths start with "stem/", so strip it
                    def _adjust(name):
                        # remove "stem/" prefix
                        return name.partition('/')[2]
                else:
                    _adjust = lambda name: name
            else:
                _adjust = lambda name: name

            # Extract each member under target_dir, adjusting its path
            for member in zip_ref.infolist():
                adjusted_path = _adjust(member.filename)
                if not adjusted_path:
                    # skip the empty name that comes from the root folder entry
                    continue
                out_path = target_dir / adjusted_path
                if member.is_dir():
                    out_path.mkdir(parents=True, exist_ok=True)
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as src, open(out_path, 'wb') as dst:
                        dst.write(src.read())

            print(f"Extracted '{self.source_zip.name}' to '{target_dir}'")

    # backward-compat alias
    extract_all = extract

