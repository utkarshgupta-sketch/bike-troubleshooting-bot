# preflight.py — architecture smoke test. Proves the risky parts before we build.
import sys, os, glob
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

RESULTS = []
def check(name, fn):
    try:
        RESULTS.append(("PASS", name, fn() or ""))
    except Exception as e:
        RESULTS.append(("FAIL", name, f"{type(e).__name__}: {e}"))

MANUALS = Path(__file__).parent / "manuals"

def t_python():
    assert sys.version_info >= (3, 11), f"need 3.11+, have {sys.version.split()[0]}"
    return sys.version.split()[0]

def t_imports():
    import streamlit, PyPDF2, sentence_transformers, faiss, langdetect, dotenv, sarvamai
    return "all packages import"

def t_parser():
    pdfs = sorted(glob.glob(str(MANUALS / "*.pdf")))
    assert pdfs, "no PDFs in manuals/"
    ok, bad = [], []
    for p in pdfs:
        (ok if len(Path(p).stem.split("_")) == 5 else bad).append(Path(p).name)
    return f"{len(ok)} parsed, skipped {bad}"

def t_pdf_text():
    import PyPDF2
    rep = []
    for p in sorted(glob.glob(str(MANUALS / "*.pdf"))):
        r = PyPDF2.PdfReader(p)
        text = "".join((pg.extract_text() or "") for pg in r.pages[:5])
        w = len(text.split())
        rep.append(f"{Path(p).name[:28]}: {w}w [{'OK' if w>50 else 'LOW?'}]")
    return " | ".join(rep)

def t_embedder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    m.encode("front brake adjustment"); m.encode("ब्रेक कैसे समायोजित करें")
    return f"dim={len(m.encode('x'))}, EN+HI encoded"

def t_faiss():
    import faiss, numpy as np
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    chunks = ["tyre pressure is 29 psi front", "oil change every 6000 km",
              "headlight bulb replacement"]
    v = np.array(m.encode(chunks)).astype("float32")
    idx = faiss.IndexFlatL2(v.shape[1]); idx.add(v)
    q = np.array([m.encode("what is the tyre pressure")]).astype("float32")
    _, I = idx.search(q, 1)
    return f"top match: '{chunks[I[0][0]]}'"

def t_langdetect():
    from langdetect import detect
    return f"EN->{detect('how do I adjust the brakes')}, HI->{detect('ब्रेक कैसे ठीक करें')}"

def t_sarvam_chat():
    from sarvamai import SarvamAI
    key = os.environ.get("SARVAM_API_KEY")
    assert key, "SARVAM_API_KEY not set (.env)"
    c = SarvamAI(api_subscription_key=key)
    r = c.chat.completions(model="sarvam-30b",
                           messages=[{"role": "user", "content": "Reply with exactly: OK"}])
    return f"sarvam-30b said: {r.choices[0].message.content[:40]}"

def t_vision():
    sample = MANUALS.parent / "test_image.jpg"
    if not sample.exists():
        return "SKIPPED (drop a test_image.jpg in project root to test Vision)"
    from sarvamai import SarvamAI
    c = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
    job = c.document_intelligence.create_job(language="en-IN", output_format="md")
    job.upload_file(str(sample)); job.start(); job.wait_until_complete()
    return f"vision job done: {str(job.download_output())[:40]}"

for n, f in [("Python 3.11+", t_python), ("Package imports", t_imports),
             ("Filename parser", t_parser), ("PyPDF2 real text", t_pdf_text),
             ("Multilingual embedder", t_embedder), ("FAISS retrieve", t_faiss),
             ("langdetect EN/HI", t_langdetect), ("Sarvam-30B chat", t_sarvam_chat),
             ("Sarvam Vision", t_vision)]:
    check(n, f)

print("\n==== PREFLIGHT REPORT ====")
for s, n, d in RESULTS:
    icon = "⏭️" if "SKIP" in d else ("✅" if s == "PASS" else "❌")
    print(f"{icon} {s:4} {n}\n     {d}")
fails = [r for r in RESULTS if r[0] == "FAIL" and "SKIP" not in r[2]]
print(f"\n{len(RESULTS)-len(fails)}/{len(RESULTS)} checks passed.")
sys.exit(1 if fails else 0)
