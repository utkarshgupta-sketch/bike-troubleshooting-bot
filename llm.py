"""
llm.py — turns retrieved passages into a grounded, cited answer using Sarvam-30B.

The system prompt enforces the four assignment requirements:
  1. answer ONLY from the provided passages,
  2. refuse politely if the answer isn't there (no fabrication),
  3. reply in the same language as the user's question,
  4. cite the page(s) used.
"""
import os
import re
import logging

from dotenv import load_dotenv

import rag

load_dotenv()
logger = logging.getLogger("llm")

CHAT_MODEL = "sarvam-30b"        # verified: NOT legacy sarvam-m
_CLIENT = None

# langdetect codes -> human language names we put in the prompt
LANG_NAMES = {"en": "English", "hi": "Hindi"}


def get_client():
    global _CLIENT
    if _CLIENT is None:
        from sarvamai import SarvamAI
        key = os.environ.get("SARVAM_API_KEY")
        if not key:
            raise RuntimeError("SARVAM_API_KEY is not set in .env")
        _CLIENT = SarvamAI(api_subscription_key=key)
    return _CLIENT


def detect_language(text: str):
    """Return (human_name, code). Defaults to English on failure/short text."""
    try:
        from langdetect import detect
        code = detect(text)
    except Exception:
        code = "en"
    return LANG_NAMES.get(code, "English"), code


SYSTEM_TEMPLATE = """You are "Bike Troubleshooting Bot", a careful assistant that \
helps motorcycle riders using ONLY the official manual excerpts given to you in each \
message.

Follow these rules strictly:

1. GROUNDING: Use ONLY the information in the numbered MANUAL PASSAGES provided by the \
user. Do not use outside knowledge, general facts, or assumptions. The manual is your \
only source of truth.

2. REFUSAL: If the answer is not clearly contained in those passages, do NOT guess or \
make anything up.

3. LANGUAGE: Write your ENTIRE answer in {language}, because that is the language of the \
user's question. Do this even if the passages are written in another language — \
translate the relevant content faithfully.

4. CITATION: When you give an answer, cite the manual page(s) you used, taken from the \
labels shown with each passage (for example: "(page 18)"). Never invent page numbers; \
only cite labels that appear with the passages.

5. STYLE: Be concise, practical and safety-conscious. If a passage contains a safety \
warning relevant to the question, include it.

6. FORMAT: If the answer is NOT clearly in the passages, reply with exactly the single \
token NOT_FOUND and nothing else. Otherwise, give your answer normally and never write \
the word NOT_FOUND. Do not repeat yourself."""


EXPAND_SYSTEM = """You turn a rider's question into search queries for a motorcycle \
manual. Output 3 short alternative search queries that would help locate the answer — \
use synonyms and specific component/section terms (e.g. "tyre pressure", "before riding \
checklist", "engine oil level"). One query per line. No numbering, no explanations. \
Write them in the same language as the question."""


def expand_query(query):
    """Return [original + up to 3 reworded search queries] to stabilise retrieval.

    Falls back to just [query] on any problem, so retrieval never breaks.
    """
    try:
        resp = get_client().chat.completions(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0, reasoning_effort="low", max_tokens=1500,
            request_options={"timeout_in_seconds": 60, "max_retries": 1},
        )
        text = resp.choices[0].message.content or ""
        extra = [ln.strip(" -•\t").strip() for ln in text.splitlines() if ln.strip()]
        extra = [e for e in extra if e][:3]
        return [query] + extra
    except Exception as e:
        logger.warning("query expansion failed (%s); using original only", e)
        return [query]


# Bilingual user-facing messages for the rephrase-before-refuse flow.
REPHRASE_PROMPT = {
    "English": ("I couldn't find that in the {name}. Could you describe the problem "
                "differently — for example the specific part, the symptom, or other words?"),
    "Hindi": ("मुझे यह {name} में नहीं मिला। क्या आप समस्या को अलग शब्दों में बता सकते हैं — "
              "जैसे कोई विशेष पुर्ज़ा, कोई लक्षण, या दूसरे शब्द?"),
}
HARD_REFUSAL = {
    "English": ("I couldn't find this in the {name}. Please consult a qualified mechanic "
                "or your dealer."),
    "Hindi": ("मुझे यह {name} में नहीं मिला। कृपया किसी योग्य मैकेनिक या अपने डीलर से "
              "परामर्श करें।"),
}


def message(kind, language, manual_name):
    """kind = 'rephrase' or 'refuse'. Returns the text in the user's language."""
    table = REPHRASE_PROMPT if kind == "rephrase" else HARD_REFUSAL
    return table.get(language, table["English"]).format(name=manual_name)


def _split_status(text):
    """Decide found vs not-found. A real answer never contains NOT_FOUND, so the
    token appearing anywhere => refusal. Returns (found, body)."""
    if re.search(r"NOT[_ ]?FOUND", text, re.IGNORECASE):
        return False, ""
    body = re.sub(r"^\s*FOUND[:\s]*", "", text, flags=re.IGNORECASE).strip()
    return True, _dedupe(body)


def _dedupe(text):
    """Drop repeated lines (safety net against any model repetition loop)."""
    seen, out = set(), []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out).strip()


def build_context(results, page_kind):
    blocks = []
    for i, r in enumerate(results, 1):
        cite = rag.page_citation(page_kind, r["page_start"], r["page_end"])
        blocks.append(f"[{i}] ({cite}) {r['text']}")
    return "\n\n".join(blocks)


def answer(query, results, manual_name, page_kind, language_override=None):
    """Call Sarvam-30B. Returns (found, body, language).

    found=False means the answer isn't in the manual (caller runs the
    rephrase-then-refuse flow). body is the answer text when found.
    """
    if language_override and language_override != "auto":
        language = language_override
    else:
        language, _ = detect_language(query)

    system = SYSTEM_TEMPLATE.format(manual_name=manual_name, language=language)

    def call(passages, max_words=None):
        if max_words:  # shorten each passage but keep ALL of them (preserve coverage)
            passages = [{**p, "text": " ".join(p["text"].split()[:max_words])}
                        for p in passages]
        user = (f"MANUAL: {manual_name}\n\n"
                f"MANUAL PASSAGES:\n{build_context(passages, page_kind)}\n\n"
                f"QUESTION: {query}")
        resp = get_client().chat.completions(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,        # deterministic => same question, same answer
            reasoning_effort="low", # minimum hidden "thinking" (cannot be disabled)
            max_tokens=4096,        # starter-tier hard cap (reasoning shares this budget)
            frequency_penalty=0.2,  # mild: discourage repetition loops without distorting facts
            request_options={"timeout_in_seconds": 120, "max_retries": 1},
        )
        return (resp.choices[0].message.content or "").strip()

    # If the model's hidden reasoning used up the 4096-token budget (empty answer),
    # retry with shortened passages — same coverage, smaller input, so it fits.
    raw = call(results)
    if not raw:
        raw = call(results, max_words=130)
    if not raw:
        # couldn't get any text — treat as "not found" so the rephrase flow runs
        return False, "", language

    found, body = _split_status(raw)
    if found and not body:           # said FOUND but gave nothing
        found = False
    return found, body, language
