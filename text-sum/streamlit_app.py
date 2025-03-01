from flask import Flask, request, jsonify, render_template_string
import mlflow
from transformers import pipeline
import logging

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def run_summarization(text, run_name="FlaskSummarization", max_length=200, min_length=100):
    """
    Summarizes the provided text using the pre-loaded summarization pipeline.
    This function logs parameters and metrics to MLflow.
    """
    try:
        logger.info(f"Starting MLflow run: {run_name}")
        mlflow.start_run(run_name=run_name)
        mlflow.log_param("model", "facebook/bart-large-cnn")
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("min_length", min_length)

        #Summarization
        result=summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary_text = result[0]['summary_text']

        mlflow.log_metric("summary_length", len(summary_text))
        logger.info(f"Summarization successful, summary length: {len(summary_text)}")
        mlflow.end_run()
        return summary_text
    except Exception as e:
        logger.error("Error during summarization", exc_info=True)
        mlflow.end_run()
        return f"Error: {str(e)}"

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>Text Summarization</title>
  </head>
  <body>
    <h1>Text Summarization</h1>
    {% if error %}
      <p style="color: red;">{{ error }}</p>
    {% endif %}
    <form method="post">
      <textarea name="text" rows="10" cols="80" placeholder="Enter your text here...">{{ text if text else '' }}</textarea><br>
      <input type="submit" value="Summarize">
    </form>
    {% if summary %}
      <h2>Summary:</h2>
      <p>{{ summary }}</p>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles both displaying the form and processing the text summarization.
    If the user submits text via the form, the text is summarized and displayed.
    """
    if request.method == "POST":
        text = request.form.get("text", "")
        if not text.strip():
            return render_template_string(HTML_TEMPLATE, error="Please enter some text to summarize.", text=text)
        summary = run_summarization(text, run_name="UI_Text_Summarization")
        return render_template_string(HTML_TEMPLATE, summary=summary, text=text)
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/summarize", methods=["POST"])
def summarize_api():
    """
    API endpoint to summarize text.
    Expects a JSON payload with a "text" field.
    :return: JSON summary
    """
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided."}), 400

    summary = run_summarization(text)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

"""
When you run app.py, the Flask application will be available on port 5000. 
Users can now access the UI via a web browser (e.g., navigating to http://localhost:5000/), 
paste their text into the form, and see the summarized result
"""