import streamlit as st
import fitz
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()


def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_bytes = uploaded_file.read()
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in pdf_document:
        text += page.get_text()

    return text


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text


def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)


def summarize_text(text, num_sentences=8):
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    top_sentence_indices = sentence_scores.argsort()[-num_sentences:]
    top_sentence_indices = sorted(top_sentence_indices)

    summary = " ".join([sentences[i] for i in top_sentence_indices])

    return summary


def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    stop_words = {
        "this", "that", "with", "from", "were", "have",
        "their", "these", "using", "paper", "study",
        "research", "which", "such", "been", "into"
    }

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    word_counts = Counter(filtered_words)

    return [word for word, count in word_counts.most_common(top_n)]


def extract_section(text, keywords, max_chars=3000):
    lower_text = text.lower()

    for keyword in keywords:
        pos = lower_text.find(keyword)
        if pos != -1:
            return text[pos:pos + max_chars]

    return "Section not clearly found in this paper."


def answer_question(question, text):
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return "No content found."

    sentence_embeddings = model.encode(sentences)
    question_embedding = model.encode([question])

    similarities = cosine_similarity(question_embedding, sentence_embeddings)[0]
    best_match_index = similarities.argmax()

    return sentences[best_match_index]


st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📘",
    layout="wide"
)

st.sidebar.title("📘 AI Research Assistant")
st.sidebar.markdown("Upload a research paper and analyze it intelligently.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Research Paper PDF",
    type=["pdf"]
)

st.title("📘 Research Paper Summarization Engine")
st.caption("AI-powered academic document analysis and question answering")

if uploaded_file is not None:
    with st.spinner("Processing research paper..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(extracted_text)

        executive_summary = summarize_text(cleaned_text)
        keywords = extract_keywords(cleaned_text)

        methodology_text = extract_section(
            cleaned_text,
            ["methodology", "methods", "approach", "experimental setup"]
        )

        limitations_text = extract_section(
            cleaned_text,
            ["limitations", "challenges", "constraints"]
        )

        future_work_text = extract_section(
            cleaned_text,
            ["future work", "future directions", "conclusion"]
        )

    st.success("Paper analyzed successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pages Processed", len(extracted_text) // 3000 + 1)

    with col2:
        st.metric("Characters Extracted", len(cleaned_text))

    with col3:
        st.metric("Q&A Ready", "Yes")

    st.subheader("Extracted Keywords")
    st.write(", ".join(keywords))

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary",
        "Methodology",
        "Limitations",
        "Future Work",
        "Q&A"
    ])

    with tab1:
        st.write(executive_summary)

    with tab2:
        st.write(methodology_text[:1200])

    with tab3:
        st.write(limitations_text[:1200])

    with tab4:
        st.write(future_work_text[:1200])

    with tab5:
        user_question = st.text_input("Ask a question about the paper")

        if user_question:
            answer = answer_question(user_question, cleaned_text)
            st.success(answer)

    with st.expander("View Extracted Raw Text"):
        st.text_area(
            "Paper Content",
            cleaned_text[:5000],
            height=400
        )