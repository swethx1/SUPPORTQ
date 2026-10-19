
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Project paths and database setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = BASE_DIR / "data" / "faqs.json"
DB_DIR = BASE_DIR / "chroma_db"

# Converts text into numerical embeddings.
model = SentenceTransformer("all-MiniLM-L6-v2")

# Stores data persistently on this computer.
client = chromadb.PersistentClient(path=str(DB_DIR))


# --------------------------------------------------
# 2. Existing FAQ collection
# --------------------------------------------------

faq_collection = client.get_or_create_collection(
    name="support_faqs",
    metadata={"hnsw:space": "cosine"},
)


# --------------------------------------------------
# 3. New collection for uploaded documents
# --------------------------------------------------

document_collection = client.get_or_create_collection(
    name="support_documents",
    metadata={"hnsw:space": "cosine"},
)


# --------------------------------------------------
# 4. Existing FAQ indexing
# --------------------------------------------------

def index_faqs():
    """Load FAQs and store their embeddings in ChromaDB."""

    with FAQ_FILE.open("r", encoding="utf-8") as file:
        faqs = json.load(file)

    documents = [
        f"Question: {faq['question']} Answer: {faq['answer']}"
        for faq in faqs
    ]

    ids = [str(faq["id"]) for faq in faqs]

    if not documents:
        return 0

    embeddings = model.encode(documents).tolist()

    # Replace existing FAQ entries safely.
    faq_collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=[
            {
                "category": faq["category"],
                "question": faq["question"],
            }
            for faq in faqs
        ],
    )

    return len(faqs)


# --------------------------------------------------
# 5. Existing FAQ search
# --------------------------------------------------

def search_similar_faqs(query: str, limit: int = 3):
    """Find FAQs whose meanings are closest to the query."""

    if limit < 1:
        return []

    total = faq_collection.count()

    if total == 0:
        return []

    query_embedding = model.encode(query).tolist()

    results = faq_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, total),
    )

    matches = []

    if not results["ids"] or not results["ids"][0]:
        return matches

    for i, faq_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        matches.append(
            {
                "id": int(faq_id),
                "category": metadata["category"],
                "question": metadata["question"],
                "content": results["documents"][0][i],
                "distance": round(distance, 4),
            }
        )

    return matches


# --------------------------------------------------
# 6. Split an uploaded document into chunks
# --------------------------------------------------

def chunk_document_text(
    extracted_text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
):
    """
    Split extracted PDF text into smaller pieces.

    Keep the page number with each chunk so that
    answers can identify their source page.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    # document_processor.py adds markers such as [Page 1].
    page_pattern = r"\[Page (\d+)\]\s*"
    parts = re.split(page_pattern, extracted_text)

    chunks = []
    step = chunk_size - overlap

    # The split produces alternating text and page numbers.
    for index in range(1, len(parts), 2):
        page_number = int(parts[index])
        page_text = parts[index + 1].strip()

        if not page_text:
            continue

        start = 0

        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "content": chunk_text,
                        "page": page_number,
                    }
                )

            if end == len(page_text):
                break

            start += step

    # Handle text without page markers, if any.
    if not chunks and extracted_text.strip():
        text = extracted_text.strip()
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "content": chunk_text,
                        "page": 1,
                    }
                )

            if end == len(text):
                break

            start += step

    return chunks


# --------------------------------------------------
# 7. Index an uploaded document
# --------------------------------------------------

def index_document_chunks(document_id: str, chunks: list):
    """
    Store a document's chunks in the document collection.

    Each chunk is tagged with its document ID and page.
    """

    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if not chunks:
        raise ValueError("No document chunks were provided.")

    contents = [
        chunk["content"]
        for chunk in chunks
        if chunk.get("content", "").strip()
    ]

    pages = [
        int(chunk["page"])
        for chunk in chunks
        if chunk.get("content", "").strip()
    ]

    if not contents:
        raise ValueError("No readable document text was provided.")

    embeddings = model.encode(contents).tolist()

    ids = [
        f"{document_id}_chunk_{index}"
        for index in range(len(contents))
    ]

    metadatas = [
        {
            "document_id": document_id,
            "page": page,
            "chunk_index": index,
        }
        for index, page in enumerate(pages)
    ]

    # Re-indexing the same document ID replaces its chunks.
    existing = document_collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        document_collection.delete(ids=existing["ids"])

    document_collection.upsert(
        ids=ids,
        documents=contents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(contents)


# --------------------------------------------------
# 8. Search within one selected document
# --------------------------------------------------

def search_document_chunks(
    document_id: str,
    query: str,
    limit: int = 4,
):
    """
    Find relevant passages from the selected document only.
    """

    if not document_id.strip():
        return []

    if not query.strip() or limit < 1:
        return []

    # First check whether this document exists.
    existing = document_collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if not existing["ids"]:
        return []

    query_embedding = model.encode(query).tolist()

    results = document_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(limit, len(existing["ids"])),
        where={"document_id": document_id},
    )

    matches = []

    if not results["ids"] or not results["ids"][0]:
        return matches

    for index, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][index]

        matches.append(
            {
                "id": chunk_id,
                "document_id": metadata["document_id"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
                "content": results["documents"][0][index],
                "distance": round(
                    results["distances"][0][index],
                    4,
                ),
            }
        )

    return matches


# --------------------------------------------------
# 9. Delete a document and its indexed chunks
# --------------------------------------------------

def delete_document(document_id: str):
    """Remove all indexed chunks belonging to one document."""

    existing = document_collection.get(
        where={"document_id": document_id}
    )

    if existing["ids"]:
        document_collection.delete(ids=existing["ids"])

    return len(existing["ids"])
