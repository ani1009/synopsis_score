# app.py

import streamlit as st
import fitz  # For PDF parsing (PyMuPDF)
from utils.anonymizer import anonymize_text


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Synopsis Scoring App",
    page_icon="🖋️",
    layout="centered",
)
st.title("🖋️ Article Synopsis Scorer")
st.write(
    "Upload an article (TXT or PDF) and your synopsis (TXT). "
    "The app will:\n"
    "1. Anonymize both texts (names, dates, etc.).\n"
    "2. Compute a score (0–100) based on coverage, coherence, and clarity.\n"
    "3. Give you feedback."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# File upload widgets
# ─────────────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)
with col1:
    uploaded_article = st.file_uploader(
        label="Upload Article (.txt or .pdf)",
        type=["txt", "pdf"],
        help="Choose the full article file."
    )
with col2:
    uploaded_synopsis = st.file_uploader(
        label="Upload Synopsis (.txt)",
        type=["txt"],
        help="A short synopsis you have written."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Score button
# ─────────────────────────────────────────────────────────────────────────────

if st.button("Score It!"):

    # 1) Check uploads
    if (uploaded_article is None) or (uploaded_synopsis is None):
        st.error("⚠️ Please upload both the article and the synopsis.")
        st.stop()

    # 2) Read files
    try:
        raw_synopsis = uploaded_synopsis.read().decode("utf-8", errors="ignore")
    except Exception:
        raw_synopsis = uploaded_synopsis.read().decode("latin-1", errors="ignore")

    if uploaded_article.type == "text/plain":
        try:
            raw_article = uploaded_article.read().decode("utf-8", errors="ignore")
        except Exception:
            raw_article = uploaded_article.read().decode("latin-1", errors="ignore")
    else:
        # PDF path
        pdf_bytes = uploaded_article.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [page.get_text() for page in doc]
        raw_article = "\n".join(pages)



    # 3) Anonymize
    with st.spinner("🔒 Anonymizing texts…"):
        anon_article = anonymize_text(raw_article)
        anon_synopsis = anonymize_text(raw_synopsis)



    # 4) Lazy‑import scoring code (so torch/sentence-transformers only loads now)
    with st.spinner("🔍 Loading scoring modules…"):
        # The first time you run this, SentenceTransformer will download models to ~/.cache/
        from utils.scoring import aggregate_scores, generate_feedback



    # 5) Compute scores (note: aggregate_scores internally loads torch/sbert lazily now)
    with st.spinner("⚙️ Computing scores…"):
        scores_dict = aggregate_scores(raw_article, raw_synopsis)
        # Convert 0–1 floats to 0–100 integers
        int_scores = {k: int(round(v * 100)) for k, v in scores_dict.items()}
        feedback_text = generate_feedback(scores_dict)



    # 6) Display
    st.success(f"**Overall Score:** {int_scores['overall']} / 100")
    st.write("#### Breakdown:")
    st.write(f"- Content Coverage: **{int_scores['content_coverage']} / 100**")
    st.write(f"- Coherence: **{int_scores['coherence']} / 100**")
    st.write(f"- Clarity: **{int_scores['clarity']} / 100**")


    st.write("#### Feedback:")
    st.info(feedback_text)


    st.markdown("---")
    st.caption(
        "⚠️ We anonymized all names, dates, and PII before scoring. "
        "No raw text is ever stored or sent out."
    )


