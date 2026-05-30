from pathlib import Path
import streamlit as st
from catalogue import list_manuals
import rag
import llm

st.set_page_config(page_title="Bike Troubleshooting Bot", page_icon="🏍️")
st.title("🏍️ Bike Troubleshooting Bot")

UPLOAD_OPTION = "⬆️  Upload my own PDF…"

# NOTE (deferred): full interface localization (whole UI switches EN/Hindi, chosen
# first) is planned as a final polish pass after the core works. See plan backlog.

manuals = list_manuals()

# --- Source picker: pre-loaded manuals OR upload your own ---
options = [UPLOAD_OPTION] + [m["label"] for m in manuals]
choice = st.selectbox("Choose a bike manual (or upload your own):", options, index=0)

active = None  # one unified "active manual" used by the rest of the app

if choice == UPLOAD_OPTION:
    st.info("🔒 Your uploaded manual is used **for this session only** — it is NOT "
            "saved to disk, NOT added to the catalogue, and NOT logged.")
    uploaded = st.file_uploader("Upload a manual PDF", type=["pdf"])
    if uploaded is not None:
        name = Path(uploaded.name).stem
        active = {
            "source": "upload",
            "filename": uploaded.name,
            "label": f"{name} (uploaded)",
            "file": uploaded,  # in-memory file, never written to disk
        }
        st.success(f"Ready: {active['label']}")
    else:
        st.caption("Upload a PDF to continue.")
else:
    selected = next(m for m in manuals if m["label"] == choice)
    active = {"source": "preloaded", **selected}
    st.caption(f"Loaded {len(manuals)} pre-loaded manual(s) from the catalogue.")

# --- Show what we've got (works for both sources) ---
if active is not None:
    with st.expander("Active manual details"):
        if active["source"] == "preloaded":
            st.write({
                "Source": "Pre-loaded",
                "Brand": active["brand"], "Model": active["model"], "Year": active["year"],
                "Document type": active["doctype"], "Language": active["language"],
                "File": active["filename"],
            })
        else:
            st.write({
                "Source": "Uploaded (session only)",
                "Manual name": active["label"],
                "File": active["filename"],
            })

    st.divider()
    st.subheader("Ask about your bike")

    # Build or load the index for this manual, cached in the session so we don't
    # rebuild on every keystroke. Same code path for pre-loaded and uploaded.
    key = f"{rag.CACHE_VERSION}|{active['source']}|{active['label']}|{active.get('filename','')}"
    if active["source"] == "upload":
        key += f"|{getattr(active['file'], 'size', '')}"

    if st.session_state.get("idx_key") != key:
        try:
            with st.spinner("Reading and indexing the manual… (first time can take a moment)"):
                mi, info = rag.get_manual_index(active)
            st.session_state.update(idx_key=key, idx=mi, idx_info=info, idx_error=None)
        except ValueError as e:
            st.session_state.update(idx_key=key, idx=None, idx_info=None, idx_error=str(e))

    if st.session_state.get("idx_error"):
        st.warning("⚠️ " + st.session_state["idx_error"] +
                   "\n\n_The automatic OCR rescue for unreadable PDFs comes online at the "
                   "upload/image step._")
    elif st.session_state.get("idx"):
        info = st.session_state["idx_info"]
        st.caption(
            f"Indexed **{info['chunks']} passages** · {info['pages']} pages · "
            f"{info['words']:,} words · source: `{info['method']}` · "
            f"page numbers: `{info['page_kind']}`"
            + (" · loaded from cache" if info.get("from_cache") else " · freshly built"))

        # A friendly manual name used in the answer + refusal message
        if active["source"] == "preloaded":
            manual_name = f"{active['brand']} {active['model']} {active['year']} {active['doctype']}"
        else:
            manual_name = Path(active["filename"]).stem

        with st.form("ask"):
            q = st.text_area("Your question (type in English or Hindi):", height=80,
                             placeholder="e.g. How often should I change the engine oil?")
            submitted = st.form_submit_button("Get answer")

        if submitted and q.strip():
            try:
                with st.spinner("Reading the manual and answering…"):
                    queries = llm.expand_query(q)                       # stabilise retrieval
                    results = st.session_state["idx"].search_multi(queries, k=6)
                    found, body, lang = llm.answer(q, results, manual_name, info["page_kind"])

                if found:
                    st.session_state["refuse_count"] = 0
                    st.markdown("### Answer")
                    st.markdown(body)
                    st.caption(f"Answered in: {lang}")
                    with st.expander("Show the manual passages used (sources)"):
                        for n, r in enumerate(results, 1):
                            cite = rag.page_citation(info["page_kind"], r["page_start"], r["page_end"])
                            st.markdown(f"**#{n} · {cite} · relevance {r['score']:.2f}**")
                            st.write(r["text"])
                else:
                    # rephrase-before-refuse: ask once, then give the firm refusal
                    cnt = st.session_state.get("refuse_count", 0) + 1
                    if cnt == 1:
                        st.session_state["refuse_count"] = 1
                        st.info("🔁 " + llm.message("rephrase", lang, manual_name))
                    else:
                        st.session_state["refuse_count"] = 0
                        st.warning(llm.message("refuse", lang, manual_name))
            except Exception as e:
                st.error(f"Couldn't get an answer from Sarvam: {e}")
        elif submitted:
            st.warning("Please type a question first.")
