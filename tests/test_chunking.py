
import pytest

from app.retriever import chunk_document_text


def test_chunks_keep_the_correct_page_number():
    text = "[Page 1]\nThis is page one.\n\n[Page 2]\nThis is page two."

    chunks = chunk_document_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 1
    assert chunks[1]["page"] == 2
    assert "This is page one." in chunks[0]["content"]
    assert "This is page two." in chunks[1]["content"]


def test_long_text_is_split_into_multiple_chunks():
    text = "[Page 1]\n" + ("A" * 25)

    chunks = chunk_document_text(text, chunk_size=10, overlap=2)

    assert len(chunks) == 3
    assert all(len(chunk["content"]) <= 10 for chunk in chunks)


def test_chunks_overlap_as_expected():
    text = "[Page 1]\n" + ("ABCDEFGHIJ" * 3)

    chunks = chunk_document_text(text, chunk_size=10, overlap=3)

    assert chunks[0]["content"][-3:] == chunks[1]["content"][:3]


def test_empty_text_returns_no_chunks():
    assert chunk_document_text("") == []
    assert chunk_document_text("   ") == []


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_document_text("some text", chunk_size=0)


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_document_text("some text", chunk_size=10, overlap=10)
