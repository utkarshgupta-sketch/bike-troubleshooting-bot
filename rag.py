"""
rag.py — the shared "search brain" used by BOTH pre-loaded and uploaded manuals.

Pipeline:  get text -> check readable -> chunk (with page tracking) ->
           embed (multilingual) -> FAISS index -> search(top-k).

Pre-loaded manuals: the FAISS index is cached to cache/ so we don't rebuild on
every run. Uploaded manuals: index lives in memory only (caller holds it).
"""
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np

logger = logging.getLogger("rag")

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
WORDS_PER_CHUNK = 300          # keeps headings with their lists; less fragmentation
CHUNK_OVERLAP = 80             # generous overlap => stable retrieval across phrasings
MIN_WORDS_PER_PAGE = 5         # below this average => PDF likely not readable
MIN_LETTERS_PER_CHUNK = 25     # drop near-empty / table-noise chunks
CACHE_VERSION = "v5"           # bump to invalidate old cached indexes

CACHE_DIR = Path(__file__).parent / "cache"
TEXT_DIR = Path(__file__).parent / "manual_text"   # OCR'd text layers (committed)

_EMBEDDER = None


def clean_text(t: str) -> str:
    """Repair common PDF/OCR extraction noise so embeddings match better."""
    if not t:
        return ""
    # strip markdown image embeds / base64 data-URIs (Sarvam OCR adds these for
    # image regions; the long base64 blobs otherwise bloat chunks enormously)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+", " ", t)
    t = unicodedata.normalize("NFKC", t)                 # ﬁ->fi, ﬀ->ff, etc.
    # drop private-use glyphs (stray bullet/symbol codes like )
    t = "".join(" " if 0xE000 <= ord(ch) <= 0xF8FF else ch for ch in t)
    # rejoin hyphenation splits: 'condi - tion' / 'exten - sively' -> one word
    t = re.sub(r"([A-Za-z])\s*-\s*([a-z])", r"\1\2", t)
    # remove runs of dashes/bullets (table rules) and dotted TOC leaders
    t = re.sub(r"(?:[-–—_·•]\s*){3,}", " ", t)
    t = re.sub(r"\.{3,}", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _letters(t: str) -> int:
    return sum(ch.isalpha() for ch in t)


_STOPWORDS = set(
    "a an the of to in on for and or is are be how do does what when where which why "
    "my your this that with without it its as at can need should i you we will would "
    "could from about into has have had not but very poor bad good".split())


def _keywords(query: str):
    """Content words from the query, for literal keyword matching."""
    toks = re.findall(r"\w+", query.lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def format_pages(page_start, page_end):
    """Human-friendly page range, e.g. '12' or '12–14' or '81-90'."""
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}–{page_end}"


def page_citation(kind, page_start, page_end):
    """Citation string honest about which kind of page number we have."""
    rng = format_pages(page_start, page_end)
    if kind == "printed":
        return f"page {rng}"
    if kind == "block":
        return f"approx. pages {rng}"
    return f"document page {rng}"   # physical PDF page (offset unknown)


def get_embedder():
    """Load the multilingual embedding model once (lazy singleton)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedder %s ...", EMBED_MODEL)
        _EMBEDDER = SentenceTransformer(EMBED_MODEL)
    return _EMBEDDER


# ---------- 1. get text, as a list of (page_label, text) ----------

def _pages_from_pdf(file_or_path):
    """Return [(physical_page:int, raw_text), ...] (1-based physical pages)."""
    import PyPDF2
    reader = PyPDF2.PdfReader(file_or_path)
    return [(i + 1, pg.extract_text() or "") for i, pg in enumerate(reader.pages)]


def _as_page_number(tok):
    """Parse a footer/header token into a page number, handling the common
    extraction quirk where the number is doubled (e.g. '143143' -> 143)."""
    if not tok.isdigit():
        return None
    s = tok
    if len(s) % 2 == 0 and s[:len(s) // 2] == s[len(s) // 2:]:
        s = s[:len(s) // 2]                    # '143143' -> '143', '167167' -> '167'
    n = int(s)
    return n if 1 <= n <= 1500 else None


def _detect_page_offset(pdf_pages_clean):
    """Find the constant offset between physical page and the manual's PRINTED
    page number, by reading lone numbers from each page's header/footer.

    pdf_pages_clean: [(physical:int, cleaned_text)]. Returns int offset or None.
    """
    from collections import Counter
    offsets = Counter()
    for phys, text in pdf_pages_clean:
        toks = text.split()
        window = toks[:4] + toks[-4:]          # header + footer region
        for t in window:
            n = _as_page_number(t)
            if n is not None:
                offsets[n - phys] += 1
    if not offsets:
        return None
    common = offsets.most_common(2)
    offset, count = common[0]
    second = common[1][1] if len(common) > 1 else 0
    # accept a clear winner: enough votes AND clearly ahead of the runner-up
    # (footer numbers extract noisily, so we trust dominance over raw count)
    if count >= 8 and count >= 1.5 * second:
        return offset
    return None


def _apply_printed_numbers(pdf_pages_clean):
    """Map physical pages to printed-page labels. Returns (pages, kind)."""
    offset = _detect_page_offset(pdf_pages_clean)
    if offset is None:
        return [(str(phys), text) for phys, text in pdf_pages_clean], "document"
    out = []
    for phys, text in pdf_pages_clean:
        printed = phys + offset
        out.append((str(printed) if printed >= 1 else str(phys), text))
    return out, "printed"


def _pages_from_ocr_text(txt_path: Path):
    """Parse our OCR file, which uses '===== PAGES a-b =====' separators."""
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"=====\s*PAGES\s+(\d+)-(\d+)\s*=====", raw)
    pages = []
    # re.split keeps captured groups: [pre, a, b, text, a, b, text, ...]
    for k in range(1, len(parts), 3):
        a, b, text = parts[k], parts[k + 1], parts[k + 2]
        pages.append((f"{a}-{b}", text))
    if not pages:  # no markers -> treat whole file as one block
        pages = [("1", raw)]
    return pages


def load_pages(active: dict):
    """Return (pages, method, page_kind). 'active' is the manual descriptor.

    Each page's text is cleaned of extraction noise. For PDFs we map physical
    pages to the manual's PRINTED page numbers (page_kind 'printed'); if that
    can't be detected reliably we keep physical pages (page_kind 'document').
    OCR'd manuals keep their 10-page block labels (page_kind 'block').
    """
    if active["source"] == "preloaded":
        stem = Path(active["filename"]).stem
        ocr_txt = TEXT_DIR / f"{stem}.txt"
        if ocr_txt.exists():
            pages = [(lbl, clean_text(t)) for lbl, t in _pages_from_ocr_text(ocr_txt)]
            return pages, "ocr-cached", "block"
        raw = _pages_from_pdf(active["path"])
    else:  # uploaded — in-memory file
        raw = _pages_from_pdf(active["file"])

    cleaned = [(phys, clean_text(text)) for phys, text in raw]
    pages, kind = _apply_printed_numbers(cleaned)
    return pages, "pypdf2", kind


def readability(pages):
    total_words = sum(len(t.split()) for _, t in pages)
    n = max(len(pages), 1)
    avg = total_words / n
    return {"pages": len(pages), "words": total_words,
            "avg_words_per_page": round(avg, 1),
            "readable": avg >= MIN_WORDS_PER_PAGE}


# ---------- 2. chunk with page tracking ----------

def chunk_pages(pages, words_per_chunk=WORDS_PER_CHUNK, overlap=CHUNK_OVERLAP):
    tagged = []  # (word, page_label)
    for label, text in pages:
        for w in text.split():
            tagged.append((w, label))
    chunks = []
    i, step = 0, max(words_per_chunk - overlap, 1)
    while i < len(tagged):
        window = tagged[i:i + words_per_chunk]
        if not window:
            break
        words = [w for w, _ in window]
        labels = [l for _, l in window]
        text = " ".join(words)
        i += step
        if _letters(text) < MIN_LETTERS_PER_CHUNK:   # skip table-noise / near-empty
            continue
        chunks.append({
            "text": text,
            "page_start": labels[0],
            "page_end": labels[-1],
        })
    return chunks


# ---------- 3. embed + FAISS ----------

def _embed(texts):
    vecs = get_embedder().encode(texts, show_progress_bar=False,
                                 convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


class ManualIndex:
    def __init__(self, chunks, index, label):
        self.chunks = chunks
        self.index = index
        self.label = label

    def _candidates(self, query, candidates=40):
        """Return {chunk_index: (semantic_score, keyword_overlap)} for a query.

        Candidates = semantic top-N UNION every chunk literally containing a
        query keyword (scored with its true similarity via index.reconstruct).
        """
        n = len(self.chunks)
        q = _embed([query])
        sims, idxs = self.index.search(q, min(candidates, n))
        sem = {int(i): float(s) for s, i in zip(sims[0], idxs[0]) if i >= 0}
        kws = set(_keywords(query))
        if kws:
            for i, c in enumerate(self.chunks):
                if i in sem:
                    continue
                if any(w in c["text"].lower() for w in kws):
                    sem[i] = float(np.dot(q[0], self.index.reconstruct(i)))
        out = {}
        for i, s in sem.items():
            tl = self.chunks[i]["text"].lower()
            kw = (sum(1 for w in kws if w in tl) / len(kws)) if kws else 0.0
            out[i] = (s, kw)
        return out

    def _rank(self, cand, k, keyword_weight):
        ranked = sorted(
            ((sem + keyword_weight * kw, sem, kw, i) for i, (sem, kw) in cand.items()),
            reverse=True)
        results = []
        for combined, sem, kw, i in ranked[:k]:
            c = self.chunks[i]
            results.append({
                "score": combined, "semantic": sem, "keyword": kw,
                "text": c["text"], "page_start": c["page_start"], "page_end": c["page_end"],
            })
        return results

    def search(self, query, k=6, keyword_weight=0.5):
        """Hybrid retrieval for a single query string."""
        return self._rank(self._candidates(query), k, keyword_weight)

    def search_multi(self, queries, k=6, keyword_weight=0.5):
        """Retrieval that uses expansions for RECALL but the original query for RANKING.

        Reworded phrasings are only used to gather candidate passages (so we don't
        miss a relevant chunk a single phrasing would skip). Every candidate is
        then scored against the user's ORIGINAL question, so tangential expansions
        (e.g. 'tyre pressure') can't push an off-topic passage to the top.
        """
        original = queries[0] if queries else ""
        cand_idx = set()
        for qq in queries:
            cand_idx.update(self._candidates(qq).keys())

        q0 = _embed([original])
        kws = set(_keywords(original))
        scored = {}
        for i in cand_idx:
            sem = float(np.dot(q0[0], self.index.reconstruct(i)))
            tl = self.chunks[i]["text"].lower()
            kw = (sum(1 for w in kws if w in tl) / len(kws)) if kws else 0.0
            scored[i] = (sem, kw)
        return self._rank(scored, k, keyword_weight)

    # ---- persistence (pre-loaded manuals only) ----
    def save(self, stem):
        import faiss
        CACHE_DIR.mkdir(exist_ok=True)
        faiss.write_index(self.index, str(CACHE_DIR / f"{stem}.{CACHE_VERSION}.faiss"))
        (CACHE_DIR / f"{stem}.{CACHE_VERSION}.chunks.json").write_text(
            json.dumps({"label": self.label, "chunks": self.chunks}, ensure_ascii=False),
            encoding="utf-8")

    @classmethod
    def load(cls, stem):
        import faiss
        fpath = CACHE_DIR / f"{stem}.{CACHE_VERSION}.faiss"
        cpath = CACHE_DIR / f"{stem}.{CACHE_VERSION}.chunks.json"
        if not (fpath.exists() and cpath.exists()):
            return None
        data = json.loads(cpath.read_text(encoding="utf-8"))
        index = faiss.read_index(str(fpath))
        return cls(data["chunks"], index, data["label"])


def build_index(chunks, label):
    import faiss
    vecs = _embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])   # cosine sim (vectors are normalized)
    index.add(vecs)
    return ManualIndex(chunks, index, label)


# ---------- top-level: get a ready-to-search index for an active manual ----------

def get_manual_index(active: dict, force_rebuild: bool = False):
    """Return (ManualIndex, info). Uses disk cache for pre-loaded manuals.

    Raises ValueError if the manual has no readable text (caller should offer OCR).
    """
    pages, method, page_kind = load_pages(active)
    info = readability(pages)
    info["method"] = method
    info["page_kind"] = page_kind
    if not info["readable"]:
        raise ValueError(
            f"This manual has no readable text (avg {info['avg_words_per_page']} "
            f"words/page across {info['pages']} pages). OCR rescue needed.")

    if active["source"] == "preloaded":
        stem = Path(active["filename"]).stem
        if not force_rebuild:
            cached = ManualIndex.load(stem)
            if cached is not None:
                info["chunks"] = len(cached.chunks)
                info["from_cache"] = True
                return cached, info
        chunks = chunk_pages(pages)
        mi = build_index(chunks, active["label"])
        mi.save(stem)
        info["chunks"] = len(chunks)
        info["from_cache"] = False
        return mi, info
    else:  # upload — never cached to disk
        chunks = chunk_pages(pages)
        mi = build_index(chunks, active["label"])
        info["chunks"] = len(chunks)
        info["from_cache"] = False
        return mi, info
