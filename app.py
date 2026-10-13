import hashlib
from io import BytesIO
import time

import streamlit as st

from preprocessing import clean_text, count_words
from utils.file_reader import extract_text
from utils.language import detect_language

st.set_page_config(
    page_title="AI Multilingual Document Summarizer",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_summarizer():
    from summarizer import MultilingualSummarizer

    return MultilingualSummarizer()


@st.cache_data(show_spinner=False, max_entries=4)
def extract_uploaded_file(file_name, file_digest, _file_bytes):
    if len(file_digest) != 64:
        raise ValueError("Could not verify the uploaded document.")
    return extract_text(
        BytesIO(_file_bytes),
        return_metadata=True,
        filename=file_name,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def cached_clean_text(text):
    return clean_text(text)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_detect_language(text):
    return detect_language(text[:20000])


st.title("🤖 AI-Based Multilingual Document Summarizer")
st.caption("NLP + Deep Learning + Multilingual Transformer")

with st.sidebar:
    st.header("Settings")
    max_words = st.slider("Target summary length (words)", 40, 250, 100, 10)
    chunk_tokens = st.slider(
        "Maximum input tokens per chunk",
        min_value=320,
        max_value=448,
        value=416,
        step=32,
        help="Larger chunks reduce model calls; chunks stay below the model input limit.",
    )
    st.info(
        "Supported input: PDF, DOCX, TXT, or pasted text. "
        "The multilingual model downloads on first use and stays loaded for this app session."
    )


tab1, tab2 = st.tabs(["📄 Upload Document", "✍️ Enter Text"])

text = ""
filename = "Pasted text"
file_size = 0
pdf_pages = 0
extracted_chars = 0
extracted_preview = ""
input_valid_for_summary = False
error_message = ""
input_identity = ""

with tab1:
    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt"],
        help="PDF, Microsoft Word, and plain-text files are supported.",
    )
    if uploaded is not None:
        filename = uploaded.name
        file_size = uploaded.size
        try:
            file_bytes = uploaded.getvalue()
            file_digest = hashlib.sha256(file_bytes).hexdigest()
            input_identity = hashlib.sha256(
                f"{filename}\0{file_digest}".encode("utf-8")
            ).hexdigest()
            text, meta = extract_uploaded_file(filename, file_digest, file_bytes)
            pdf_pages = meta.get("pages", 0)
            extracted_chars = meta.get("characters", 0)
            extracted_preview = text[:1000]
            input_valid_for_summary = bool(text)
            st.success(f"Extracted text from {uploaded.name}")
            st.caption(f"Pages: {pdf_pages} • Extracted characters: {extracted_chars:,}")
            with st.expander("Preview extracted text"):
                st.text_area(
                    "Extracted text preview",
                    value=text[:20000],
                    height=260,
                    disabled=True,
                )
        except (ValueError, OSError) as exc:
            error_message = str(exc)
            st.error(f"Could not read the document: {exc}")
            text = ""
            input_valid_for_summary = False
        except Exception as exc:
            error_message = str(exc)
            st.error(f"Could not read the document due to an unexpected error: {exc}")
            text = ""
            input_valid_for_summary = False

with tab2:
    pasted = st.text_area(
        "Paste your document here",
        height=320,
        placeholder="Paste an article, report, notes, research text, etc.",
    )
    if pasted.strip():
        text = cached_clean_text(pasted)
        input_identity = hashlib.sha256(pasted.encode("utf-8")).hexdigest()
        filename = "Pasted text"
        file_size = len(pasted.encode("utf-8"))
        extracted_chars = len(text)
        extracted_preview = text[:1000]
        input_valid_for_summary = bool(text)

if text:
    cleaned = text
    detected = cached_detect_language(cleaned)
    original_words = count_words(cleaned)

    c1, c2, c3 = st.columns(3)
    c1.metric("Original words", f"{original_words:,}")
    c2.metric("Detected language", detected)
    c3.metric("Characters", f"{len(cleaned):,}")

    summary_key = f"{input_identity}\0{max_words}\0{chunk_tokens}"
    if original_words < 30:
        st.warning("Please provide at least about 30 words for a meaningful summary.")

    if st.button(
        "✨ Generate Summary",
        type="primary",
        use_container_width=True,
        disabled=original_words < 30,
    ):
        started = time.perf_counter()
        progress = None
        st.session_state.pop("summary_result", None)
        try:
            if len(input_identity) != 64:
                raise ValueError("Could not verify the document before summarizing.")
            with st.spinner("Loading AI model..."):
                summarizer = load_summarizer()

            progress = st.progress(0, text="Preparing document...")
            with st.spinner("Generating summary..."):
                summary, stats = summarizer.summarize_document(
                    cleaned,
                    target_words=max_words,
                    max_input_tokens=chunk_tokens,
                    progress_callback=lambda value: progress.progress(
                        min(99, max(1, int(value))),
                        text=f"Summarizing... {int(value)}%",
                    ),
                )
            elapsed = time.perf_counter() - started
            progress.progress(100, text="Summary complete.")

            if not summary or not summary.strip():
                st.warning(
                    "The document did not produce a valid summary. "
                    "Please check the extracted text or use a longer input."
                )
            else:
                st.session_state["summary_result"] = {
                    "key": summary_key,
                    "summary": summary,
                    "stats": {**stats, "processing_seconds": elapsed},
                }
        except Exception as exc:
            if progress is not None:
                progress.empty()
            st.error(f"Summarization failed: {type(exc).__name__}: {exc}")

    result = st.session_state.get("summary_result")
    if result and result.get("key") == summary_key:
        summary = result["summary"]
        stats = result["stats"]
        st.subheader("📌 Generated Summary")
        st.write(summary)

        a, b, c, d = st.columns(4)
        a.metric("Summary words", f"{stats['summary_words']:,}")
        b.metric("Reduction", f"{stats['reduction_percent']:.1f}%")
        c.metric("Chunks processed", stats["chunks"])
        d.metric("Processing time", f"{stats['processing_seconds']:.2f} s")

        st.download_button(
            "⬇️ Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

        with st.expander("Processing details"):
            st.write({
                "file": filename,
                "language": detected,
                "original_words": stats["original_words"],
                "summary_words": stats["summary_words"],
                "reduction_percent": round(stats["reduction_percent"], 2),
                "chunks": stats["chunks"],
                "processing_seconds": round(stats["processing_seconds"], 2),
            })
else:
    if not error_message:
        st.info("Upload a document or paste text to begin.")

with st.expander("🔎 Debugging details", expanded=False):
    st.write({
        "filename": filename,
        "file_size_bytes": file_size,
        "pdf_pages": pdf_pages,
        "extracted_characters": extracted_chars,
        "first_1000_chars": extracted_preview,
        "summarizer_received_valid_input": input_valid_for_summary,
        "error_message": error_message,
    })

st.divider()
st.caption("B.Tech Project • AI-Based Multilingual Document Summarization")
