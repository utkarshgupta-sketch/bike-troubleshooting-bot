"""
catalogue.py — scans the manuals/ folder and turns each correctly-named PDF into
a catalogue entry for the dropdown.

Filename convention (exactly 5 fields, split on '_'):
    <Brand>_<Model>_<Year>_<DocType>_<Language>.pdf
Spaces inside a field are fine, e.g.:
    Royal Enfield_Bullet 350 Dual-channel_2025_Owners Manual_English.pdf

Files that do NOT have exactly 5 fields are logged as a warning and skipped —
the app never crashes on a bad filename.
"""
import logging
from pathlib import Path

logger = logging.getLogger("catalogue")

FIELDS = ("brand", "model", "year", "doctype", "language")
MANUALS_DIR = Path(__file__).parent / "manuals"


def parse_filename(pdf_path: Path):
    """Return a dict of the 5 fields, or None if the filename doesn't match."""
    stem = pdf_path.stem  # filename without .pdf
    parts = stem.split("_")
    if len(parts) != 5:
        logger.warning(
            "Skipping '%s': expected 5 underscore-separated fields, found %d.",
            pdf_path.name, len(parts),
        )
        return None
    if any(not p.strip() for p in parts):
        logger.warning("Skipping '%s': one or more fields are empty.", pdf_path.name)
        return None
    return dict(zip(FIELDS, (p.strip() for p in parts)))


def make_label(fields: dict) -> str:
    """Clean, human-readable dropdown label."""
    return (f"{fields['brand']} · {fields['model']} · {fields['year']} · "
            f"{fields['doctype']} ({fields['language']})")


def list_manuals(manuals_dir: Path = MANUALS_DIR):
    """Return a list of catalogue entries, one per valid manual, sorted by label.

    Each entry: {path, filename, label, brand, model, year, doctype, language}
    """
    entries = []
    if not manuals_dir.exists():
        logger.warning("Manuals folder not found: %s", manuals_dir)
        return entries
    for pdf in sorted(manuals_dir.glob("*.pdf")):
        fields = parse_filename(pdf)
        if fields is None:
            continue
        entries.append({
            "path": pdf,
            "filename": pdf.name,
            "label": make_label(fields),
            **fields,
        })
    entries.sort(key=lambda e: e["label"].lower())
    return entries


if __name__ == "__main__":
    # quick manual test: python catalogue.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    found = list_manuals()
    print(f"Found {len(found)} valid manual(s):")
    for e in found:
        print(f"  - {e['label']}   [{e['filename']}]")
