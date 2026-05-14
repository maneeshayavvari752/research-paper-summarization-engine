# AI Research Paper Assistant

An AI-powered research paper analysis tool that allows users to upload academic PDF documents, generate structured summaries, and ask semantic questions about the paper.

## Features

- Upload research paper PDFs
- Extract full document text
- Executive summary generation
- Methodology extraction
- Limitations detection
- Future work extraction
- Semantic question answering
- Interactive Streamlit UI
- Lightweight local inference (no paid APIs)

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- scikit-learn
- Sentence Transformers
- NumPy

## Architecture

```text
User Uploads PDF
      ↓
PyMuPDF Text Extraction
      ↓
Text Cleaning
      ↓
TF-IDF Summarization
      ↓
Section Extraction Logic
      ↓
SentenceTransformer Embeddings
      ↓
Cosine Similarity Q&A



## demo video link
https://drive.google.com/file/d/14utxm0Kw_bByqUcme8oIBnjDx59VOImo/view?usp=sharing