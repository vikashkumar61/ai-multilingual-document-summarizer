# AI-Based Multilingual Document Summarizer

A B.Tech final-year project that summarizes long documents using NLP and a multilingual Transformer model.

## Features

- PDF, DOCX and TXT upload
- Paste text directly
- Automatic language detection
- Multilingual abstractive summarization
- Long-document chunking
- Summary length control
- Word-count and reduction statistics
- Download generated summary
- CPU/GPU support

## Technology

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- mT5 multilingual XLSum
- PyMuPDF
- python-docx
- langdetect

## Installation

Use Python 3.13 for the tested setup. PyTorch, Transformers, and SentencePiece
are version-bounded in `requirements.txt` to keep their APIs and Windows wheels
compatible with this application.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

Windows:

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Linux/macOS:

```bash
.venv/bin/python -m streamlit run app.py
```

The first run downloads the Transformer model, so internet access is required once.

## Host on Streamlit Community Cloud

1. Push this project to a GitHub repository. Keep `.venv` and `venv` out of the repository; they are already ignored by `.gitignore`.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from that repository and set the app file to `app.py`.
3. Select Python 3.13 for the app environment, then deploy. Cloud builds install the packages from `requirements.txt`.

The model downloads from Hugging Face the first time the app starts, so allow extra time for the initial build and startup. Summarization runs on CPU when no CUDA GPU is available and may be slow for long documents. Choose a hosting plan with enough memory for the Transformer model; the model is loaded only when a user requests a summary.

## Project workflow

1. Input document
2. Text extraction
3. Text cleaning
4. Language detection
5. Token-bounded chunking with repeated PDF headers and page numbers removed
6. Multilingual Transformer summarization
7. Hierarchical compression for long-document chunk summaries
8. Summary statistics
9. Download summary

Text extraction and language detection are cached across Streamlit reruns. The
Transformer tokenizer and model are loaded lazily on the first summary request
and retained with `st.cache_resource`; the app shows separate loading and
generation indicators. Short documents that fit the model input limit are
summarized in one call. Longer documents are split into token-bounded chunks,
summarized, and combined into a final summary. CPU generation uses small batches,
two-beam deterministic decoding, and bounded input/output lengths; CUDA is used
with half-precision weights and batches of up to four chunks when available.
The app reports elapsed processing time, including model loading on the first
request.

## Notes

The model can run on CPU, but a CUDA-compatible GPU is strongly recommended for faster inference.

For scanned/image-only PDFs, OCR is not included in this version. Add OCR such as Tesseract in a future version.

The ROUGE scores in the academic report should be generated from a real test dataset with reference summaries; they should not be hard-coded into the application.
