import pytest
from app import app

@pytest.fixture
def client():
    # Configure the Flask app for testing.
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """
    Test that the main page loads and contains the expected title.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert b"Text Summarization" in response.data

def test_api_no_text(client):
    """
    Test the API endpoint with empty text to ensure it returns a 400 error.
    """
    response = client.post("/api/summarize", json={"text": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_api_valid_text(client):
    """
    Test the API endpoint with valid text to ensure it returns a summary.
    """
    response = client.post("/api/summarize", json={"text": "This is a test input for summarization."})
    assert response.status_code == 200
    data = response.get_json()
    assert "summary" in data
    assert isinstance(data["summary"], str)

print(app.url_map)