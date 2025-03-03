import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2
import textwrap
from rouge_score import rouge_scorer  # For evaluation


# Load summarization model (DistilBART or LongT5)
@st.cache_resource()
def load_model():
    model_name = "sshleifer/distilbart-cnn-6-6"  # or "google/long-t5-tglobal-base"
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
MAX_INPUT_TOKENS = 1024  # DistilBART token limit


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


# Function to compute ROUGE score
def compute_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores


# Preloaded test samples
sample_texts = {
    "News Article": "The government has announced new economic policies to boost small businesses. The new measures include tax relief and increased grants for startups. Analysts predict that these changes will improve the economy and create more jobs.",
    "Scientific Abstract": "Deep learning models have achieved remarkable success in many applications. This paper explores the use of neural networks for medical imaging. Our experiments show significant improvements in disease detection accuracy.",
    "Book Passage": "The protagonist, a young detective, uncovered a secret hidden within an old library. As she turned the fragile pages, a letter fell out, revealing a long-lost mystery that changed the course of her investigation."
}

# Streamlit UI
st.title("📖 AI-Powered Text & PDF Summarizer")
st.write("Summarize large documents, including PDFs and long-form text, using AI.")

# Input options: Text area, file upload, or preloaded samples
input_text = st.text_area("Paste your text here:", height=200)
uploaded_file = st.file_uploader("Or upload a PDF:", type=["pdf"])
sample_choice = st.selectbox("Or choose a preloaded sample:", ["None"] + list(sample_texts.keys()))

if sample_choice != "None":
    input_text = sample_texts[sample_choice]
    st.write("### Preloaded Sample Text:")
    st.text_area("Selected Sample", input_text, height=150, disabled=True)

if uploaded_file:
    input_text = extract_text_from_pdf(uploaded_file)
    st.write("Extracted text:")
    st.text_area("Extracted Content", input_text, height=200)

# Summarization button
if st.button("Summarize Text"):
    if not input_text:
        st.warning("Please provide text or upload a PDF.")
    else:
        st.write("Summarizing... ⏳")
        chunks = chunk_text(input_text, summarizer.tokenizer)
        summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]["summary_text"] for chunk in
                     chunks]
        final_summary = " ".join(summaries)
        st.subheader("📌 Summary:")
        st.write(final_summary)

        # Provide download option
        st.download_button("Download Summary", final_summary, file_name="summary.txt")

        # Human Feedback UI
        st.subheader("📊 Rate the Summary Quality")
        feedback = st.radio("How would you rate this summary?", ["Excellent", "Good", "Average", "Poor"])
        st.write("Your feedback helps improve the model! 👍")

        # ROUGE Score Evaluation (if reference provided)
        reference_text = st.text_area("Optional: Paste reference summary for evaluation")
        if reference_text and st.button("Evaluate with ROUGE"):
            scores = compute_rouge(reference_text, final_summary)
            st.write("🔹 ROUGE Scores:")
            st.write(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            st.write(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            st.write(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")