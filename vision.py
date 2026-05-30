"""
vision.py — image understanding via Sarvam Vision (Document Intelligence).

For an uploaded image, Sarvam Vision returns any readable TEXT *and* an inline
caption describing the image (e.g. "The image displays a sight glass…"). We use
both together as the "image context" that gets merged with the user's question.

This is the image-input feature: dashboard warning text, caution/warning labels,
a photographed manual page, or a photo of a part/symptom.
"""
import os
import glob
import html
import re
import zipfile
import tempfile
import logging
from pathlib import Path

from llm import get_client   # reuse the same Sarvam client singleton

logger = logging.getLogger("vision")


def _text_from_zip(zip_path, workdir):
    ex = os.path.join(workdir, "unzipped")
    os.makedirs(ex, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ex)

    def read_all(pattern):
        parts = []
        for p in sorted(glob.glob(os.path.join(ex, "**", pattern), recursive=True)):
            parts.append(Path(p).read_text(encoding="utf-8", errors="ignore"))
        return "\n".join(parts).strip()

    for pat in ("*.md", "*.txt"):
        t = read_all(pat)
        if t:
            return t
    t = read_all("*.html")
    if t:
        return html.unescape(re.sub(r"<[^>]+>", " ", t)).strip()
    return read_all("*.json")


def _clean(text):
    """Strip the base64 image embed Sarvam adds, plus markdown emphasis markers,
    leaving the readable text + caption."""
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)                     # ![alt](...)
    text = re.sub(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", " ", text)
    text = text.replace("*", " ").replace("#", " ")                       # md emphasis
    text = re.sub(r"\s+", " ", text).strip()
    return text[:2000]                                                    # safety cap


def understand_image(uploaded_file, language="en-IN"):
    """Return Sarvam Vision's understanding of an image: readable text + caption.

    Accepts a Streamlit UploadedFile or a file path. Returns "" on failure.
    """
    work = tempfile.mkdtemp()
    if hasattr(uploaded_file, "getvalue"):
        ext = Path(getattr(uploaded_file, "name", "image.png")).suffix or ".png"
        img_path = os.path.join(work, f"image{ext}")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getvalue())
    else:
        img_path = str(uploaded_file)

    client = get_client()
    job = client.document_intelligence.create_job(language=language, output_format="md")
    job.upload_file(img_path)
    job.start()
    job.wait_until_complete(poll_interval=5, timeout=180)
    out_zip = os.path.join(work, "out.zip")
    job.download_output(out_zip)
    text = _clean(_text_from_zip(out_zip, work))
    logger.info("Sarvam Vision understood %d chars from image", len(text))
    return text
