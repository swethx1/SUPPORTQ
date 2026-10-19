
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_search_rejects_empty_query():
    response = client.get("/search", params={"q": ""})

    assert response.status_code == 422


def test_upload_rejects_non_pdf_file():
    response = client.post(
        "/upload-document",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please upload a PDF file."


def test_upload_rejects_invalid_pdf_content():
    response = client.post(
        "/upload-document",
        files={"file": ("fake.pdf", b"this is not a PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert "does not appear to be a PDF" in response.json()["detail"]


def test_question_rejects_blank_document_id():
    response = client.post(
        "/ask-document",
        json={"document_id": "   ", "question": "What is this about?"},
    )

    assert response.status_code == 400


def test_question_rejects_blank_question():
    response = client.post(
        "/ask-document",
        json={"document_id": "some-document-id", "question": "   "},
    )

    assert response.status_code == 400


def test_question_returns_404_for_unknown_document():
    response = client.post(
        "/ask-document",
        json={
            "document_id": "test-document-that-does-not-exist",
            "question": "What is this about?",
        },
    )

    assert response.status_code == 404
