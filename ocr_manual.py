"""
ocr_manual.py — one-time tool to OCR a PDF that has no readable text layer,
using Sarvam Vision (Document Intelligence). Saves the extracted text to cache/
so we never have to OCR it again.

Why this exists: some manuals (e.g. the Hindi Honda manual) store their text as
vector drawings, so PyPDF2 reads nothing. Sarvam Vision can OCR them.

Sarvam limits (verified): max 10 pages per job, 10 requests/minute. So we split
the PDF into 10-page slices and OCR each slice as its own job, pacing ourselves.

Usage:
  # Test ONE slice (cheap) — slice index is 0-based, each slice = 10 pages:
  python ocr_manual.py "manuals/<file>.pdf" --only 6

  # Full run (all slices), saves combined text to cache/<stem>.ocr.txt:
  python ocr_manual.py "manuals/<file>.pdf"
"""
import sys, os, glob, time, zipfile, tempfile, html, re
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from sarvamai import SarvamAI

load_dotenv()

PAGES_PER_JOB = 10        # Sarvam hard limit
POLL_INTERVAL = 10.0      # seconds — keeps us under 10 requests/minute
LANGUAGE = "hi-IN"        # change per manual if needed
OUTPUT_FORMAT = "md"      # markdown; we'll also fall back to html/json/txt in the zip


def split_into_slices(src_pdf, workdir):
    reader = PdfReader(src_pdf)
    n = len(reader.pages)
    slices = []
    for s in range(0, n, PAGES_PER_JOB):
        e = min(s + PAGES_PER_JOB, n)
        writer = PdfWriter()
        for i in range(s, e):
            writer.add_page(reader.pages[i])
        out = os.path.join(workdir, f"slice_{s // PAGES_PER_JOB:03d}.pdf")
        with open(out, "wb") as f:
            writer.write(f)
        slices.append({"idx": s // PAGES_PER_JOB, "first": s + 1, "last": e, "path": out})
    return n, slices


def text_from_zip(zip_path, workdir):
    """Pull readable text out of Sarvam's output ZIP, trying md -> txt -> html -> json."""
    ex = os.path.join(workdir, "unzipped")
    os.makedirs(ex, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        z.extractall(ex)
    def read_all(pattern):
        out = []
        for p in sorted(glob.glob(os.path.join(ex, "**", pattern), recursive=True)):
            out.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(out).strip()
    for pat in ("*.md", "*.txt"):
        t = read_all(pat)
        if t:
            return t, names
    t = read_all("*.html")
    if t:
        return html.unescape(re.sub(r"<[^>]+>", " ", t)).strip(), names
    # last resort: dump any json text
    t = read_all("*.json")
    return t, names


def ocr_slice(client, pdf_path, out_zip):
    job = client.document_intelligence.create_job(language=LANGUAGE, output_format=OUTPUT_FORMAT)
    job.upload_file(pdf_path)
    job.start()
    job.wait_until_complete(poll_interval=POLL_INTERVAL, timeout=900)
    return job.download_output(out_zip)


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python ocr_manual.py <pdf> [--only <sliceIndex>]")
        sys.exit(2)
    src = args[0]
    only = None
    if "--only" in args:
        only = int(args[args.index("--only") + 1])

    assert os.environ.get("SARVAM_API_KEY"), "SARVAM_API_KEY not set in .env"
    client = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])

    stem = Path(src).stem
    cache_dir = Path("cache"); cache_dir.mkdir(exist_ok=True)
    # OCR'd text costs money to produce, so it lives in a COMMITTED folder
    # (not cache/, which is git-ignored) — preserved in the repo, reused forever.
    text_dir = Path("manual_text"); text_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as work:
        total_pages, slices = split_into_slices(src, work)
        print(f"'{stem}': {total_pages} pages -> {len(slices)} slices of <= {PAGES_PER_JOB} pages")

        todo = [s for s in slices if (only is None or s["idx"] == only)]
        if not todo:
            print(f"No slice with index {only}. Valid: 0..{len(slices)-1}")
            sys.exit(2)

        combined = []
        for s in todo:
            tag = f"slice {s['idx']} (pages {s['first']}-{s['last']})"
            print(f"  OCR {tag} ...", flush=True)
            out_zip = os.path.join(work, f"out_{s['idx']:03d}.zip")
            try:
                ocr_slice(client, s["path"], out_zip)
                text, names = text_from_zip(out_zip, os.path.join(work, f"z{s['idx']}"))
                words = len(text.split())
                print(f"    -> {words} words. zip contents: {names}")
                combined.append(f"\n\n===== PAGES {s['first']}-{s['last']} =====\n\n{text}")
                if only is not None:
                    preview = text.strip()[:600]
                    print("\n----- PREVIEW (first 600 chars) -----\n" + preview + "\n-------------------------------------")
            except Exception as ex:
                print(f"    !! FAILED on {tag}: {type(ex).__name__}: {ex}")
                raise
            time.sleep(6)  # extra spacing to respect 10 req/min across slices

        if only is None:
            out_txt = text_dir / f"{stem}.txt"
            out_txt.write_text("".join(combined), encoding="utf-8")
            print(f"\nSaved combined OCR text -> {out_txt}  ({len(''.join(combined).split())} words)")
        else:
            out_txt = cache_dir / f"{stem}.slice{only}.ocr.txt"
            out_txt.write_text("".join(combined), encoding="utf-8")
            print(f"\nSaved test slice -> {out_txt}")


if __name__ == "__main__":
    main()
