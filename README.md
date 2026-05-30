# 🏍️ Bike Troubleshooting Bot

A Streamlit web app that answers motorcycle questions **strictly from the bike's
official manual** — in **English or Hindi**, from **typed questions and/or photos** —
and **politely refuses** when the answer isn't in the manual. Built on Sarvam AI
(Sarvam‑30B + Sarvam Vision) with local retrieval.

The single most important behaviour is **document‑grounded answering with reliable
refusal**: the bot never invents an answer; it only uses the retrieved manual passages
and cites the page(s) it used.

---

## What it does

- **Pick a manual** from a dropdown (auto‑built by scanning the `manuals/` folder) **or
  upload your own PDF** (used in‑session only — never saved, catalogued, or logged).
- **Ask by text, by image, or both.** A photo (dashboard light, warning label, a part,
  a manual page) is read by Sarvam Vision (text + a description) and merged into the
  question.
- **Grounded, cited answers.** Answers come only from the manual, with page citations.
- **Rephrase‑before‑refuse.** If it can't find something, it first asks you to reword
  the question; only if that also fails does it give the firm
  "consult a qualified mechanic or your dealer" refusal.
- **Multilingual.** Ask in English or Hindi; the answer comes back in the **language of
  your question**, even if the manual is in the other language.

---

## How it works (the pipeline)

```
 Dropdown manual ─┐
                  ├─► rag.py: extract text → clean → chunk (with page tracking)
 Uploaded PDF  ───┘            → embed (multilingual) → FAISS index
                                       │  (disk‑cached for pre‑loaded manuals,
                                       │   in‑memory for uploads)
 Typed question ─┐                     ▼
 Image ─► Sarvam Vision (text+caption) → combined query
                  │                    │
                  │   query expansion (Sarvam‑30B rewrites into search angles)
                  │                    ▼
                  └──────────► hybrid retrieval (semantic + keyword), ranked
                                       │   by the typed question
                                       ▼
                               Sarvam‑30B (llm.py)
                               • answer ONLY from passages
                               • refuse if absent (NOT_FOUND)
                               • reply in the question's language
                               • cite the page(s)
                                       ▼
                               grounded, cited answer
```

**Key design choices**

- **Grounded refusal** — a strict system prompt forces answers from the provided
  passages only; the model tags replies `NOT_FOUND` when the answer isn't present
  (detected language‑independently), driving the rephrase‑then‑refuse flow.
- **Hybrid retrieval** — semantic similarity **plus** literal keyword matching, so
  exact terms (e.g. "exhaust") are reliably found, not just paraphrase matches.
- **Query expansion** — the question is rewritten into several search angles for recall,
  but candidates are **ranked by the original question** for precision (so tangents and
  verbose image captions don't hijack the results).
- **Honest page citations** — for digital PDFs we detect the manual's **printed** page
  number (handling cover/intro offsets and doubled‑number OCR quirks); when we can't,
  we label it "document page N" rather than mislead.
- **OCR rescue for unreadable PDFs** — some manuals (here, the Hindi Honda manual) store
  text as vector outlines, so PyPDF2 reads nothing. We OCR those with Sarvam Vision once
  and cache the text in `manual_text/` (committed, so it's never re‑paid for).

---

## Tech stack

- **Python 3.11+**, **Streamlit** (UI)
- **Sarvam AI** via `sarvamai`:
  - **Sarvam‑30B** for chat completion (grounded answers, query expansion)
  - **Sarvam Vision** (Document Intelligence) for image understanding and for OCR of
    unreadable PDFs
- **PyPDF2** — fast, free text extraction for digital‑native manuals
- **sentence‑transformers** (`paraphrase-multilingual-MiniLM-L12-v2`) — multilingual
  embeddings (English + Hindi, cross‑language)
- **FAISS** (`faiss-cpu`) — in‑process vector store
- **langdetect** — language of the user's question
- **python‑dotenv** — local API key handling

---

## Project structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI; ties everything together |
| `catalogue.py` | Scans `manuals/`, parses filenames, builds the dropdown |
| `rag.py` | Extract → clean → chunk → embed → FAISS → hybrid retrieve (shared core) |
| `llm.py` | System prompt + Sarvam‑30B calls (answers, refusal, query expansion) |
| `vision.py` | Sarvam Vision image understanding (text + caption) |
| `ocr_manual.py` | One‑time tool: OCR an unreadable PDF into `manual_text/` |
| `preflight.py` | One‑shot smoke test of every dependency before building |
| `manuals/` | Pre‑loaded manual PDFs |
| `manual_text/` | OCR'd text for manuals PyPDF2 can't read (committed) |
| `cache/` | Cached FAISS indexes (git‑ignored; rebuilt on demand) |

**Manual filename convention** (5 fields, split on `_`):
`<Brand>_<Model>_<Year>_<DocType>_<Language>.pdf`
e.g. `Royal Enfield_Bullet 350 Dual-channel_2025_Owners Manual_English.pdf`. Files that
don't match are logged and skipped — never crash the app.

---

## Run locally

```bash
# 1. Python 3.11+ and a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Sarvam API key
echo "SARVAM_API_KEY=your_key_here" > .env     # .env is git‑ignored

# 4. (optional) Sanity‑check everything
python preflight.py

# 5. Run
streamlit run app.py
```

Open http://localhost:8501.

> First run downloads the embedding model (~one‑time) and builds each manual's index the
> first time it's selected (cached afterwards).

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On https://share.streamlit.io, create an app pointing at `app.py`.
3. In **App settings → Secrets**, add:
   ```toml
   SARVAM_API_KEY = "your_key_here"
   ```
   (The app reads the key from `st.secrets` on Cloud and from `.env` locally.)
4. Deploy. Indexes build on first use of each manual.

`requirements.txt` pins `faiss-cpu` (not `faiss`) so the Cloud build succeeds.

---

## Multilingual support

- Works end‑to‑end in **English and Hindi**. The answer language follows the **question**
  (Hindi question → Hindi answer), including the **cross‑language** case (English manual +
  Hindi question → Hindi answer), thanks to the multilingual embedder and Sarvam‑30B.
- **Extensible by config, not code**: add a manual with the appropriate `_<Language>`
  suffix and (if needed) its language code — no code changes for new Indic languages.

---

## Known limitations & honest notes

- **Tables (maintenance/spec schedules).** PyPDF2 flattens 2‑D tables, losing which value
  aligns to which column, so table‑derived answers (e.g. exact service intervals) can be
  imperfect. **Verified fix available:** Sarvam Vision parses tables into structured
  HTML/Markdown (we confirmed it recovers a maintenance schedule exactly). We chose not to
  route all extraction through Vision (credit + latency cost) and flag this instead.
- **Same question, English vs Hindi manual can differ.** The English and Hindi manuals are
  **separate PDFs extracted by different methods** (PyPDF2 vs Vision OCR). For table‑based
  facts, retrieval can surface different fragments. The bot is faithful to what it
  retrieves — this is a retrieval‑on‑tables effect, **not** a mistranslation.
- **Retrieval variance on terse questions.** Short, vague questions whose answer is a
  one‑line table/checklist entry occasionally trigger a rephrase. This is the ceiling of
  the (assignment‑locked) embedding model plus the API tier's 4096‑token response cap;
  the rephrase‑before‑refuse flow is the cushion.
- **Image input reads text + describes; it doesn't diagnose.** Sarvam Vision returns any
  text in the image plus a caption. (We briefly trialled Gemini for richer captions but
  reverted to Sarvam‑only to stay on the specified stack.)
- **Uploaded manuals are session‑only** — not saved, catalogued, or logged, by design.

---

## Credits

Built with the Sarvam AI platform (Sarvam‑30B, Sarvam Vision) and open‑source tooling
(Streamlit, sentence‑transformers, FAISS, PyPDF2).
