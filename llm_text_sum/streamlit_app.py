import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import textwrap

# Load summarization model
@st.cache_resource()
def load_model():
    model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to chunk long text
MAX_INPUT_TOKENS = 1024

def chunk_text(text, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + sentence)) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Streamlit UI
st.title("📖 AI-Powered Text & PDF Summarizer")
st.write("Summarize large documents, including PDFs and long-form text, using AI.")

# Input options: Text area or file upload
input_text = st.text_area("Paste your text here:", height=200)
uploaded_file = st.file_uploader("Or upload a PDF:", type=["pdf"])

if uploaded_file:
    input_text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted Text:")
    st.text_area("Extracted Content", input_text, height=200)

# Summarization button
if st.button("Summarize Text"):
    if not input_text:
        st.warning("Please provide text or upload a PDF.")
    else:
        st.write("Summarizing... ⏳")
        chunks = chunk_text(input_text, summarizer.tokenizer)
        summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
        final_summary = " ".join(summaries)
        st.subheader("📌 Summary:")
        st.write(final_summary)

        # Provide download option
        st.download_button("Download Summary", final_summary, file_name="summary.txt")
