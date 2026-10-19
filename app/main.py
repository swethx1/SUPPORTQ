
import json
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.retriever import (
    index_faqs,
    search_similar_faqs,
    chunk_document_text,
    index_document_chunks,
    search_document_chunks,
)
from app.llm import generate_answer
from app.document_processor import extract_text_from_pdf


app = FastAPI(
    title="SupportIQ API",
    description="Document-based AI customer support using RAG and Groq",
    version="0.4.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
FAQ_FILE = BASE_DIR / "data" / "faqs.json"

MAX_PDF_SIZE = 10 * 1024 * 1024  # 10 MB


class ChatRequest(BaseModel):
    question: str


class DocumentQuestion(BaseModel):
    document_id: str
    question: str


def load_faqs():
    with FAQ_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/")
def home():
    return {
        "message": "Welcome to SupportIQ!",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/faqs")
def get_faqs():
    faqs = load_faqs()
    return {"count": len(faqs), "faqs": faqs}


@app.get("/search")
def search_faqs(q: str = Query(..., min_length=1)):
    faqs = load_faqs()
    query_words = q.lower().split()

    results = [
        faq
        for faq in faqs
        if any(
            word in (faq["question"] + " " + faq["answer"]).lower()
            for word in query_words
        )
    ]

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.on_event("startup")
def initialize_knowledge_base():
    index_faqs()


@app.get("/search-semantic")
def semantic_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=3, ge=1, le=10),
):
    results = search_similar_faqs(q.strip(), limit=limit)

    return {
        "query": q,
        "count": len(results),
        "results": results,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        matches = search_similar_faqs(question, limit=3)

        if not matches:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the support FAQs."
                ),
                "sources": [],
            }

        context = "\n\n".join(
            match["content"] for match in matches
        )

        answer = generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "category": match["category"],
                    "question": match["question"],
                }
                for match in matches
            ],
        }

    except Exception as exc:
        print(
            f"Chat generation error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=502,
            detail="Chat generation failed. Check the VS Code terminal.",
        ) from exc


@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    temp_path = None

    try:
        content = await file.read(MAX_PDF_SIZE + 1)

        if len(content) > MAX_PDF_SIZE:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 10 MB upload limit.",
            )

        if not content.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not appear to be a PDF.",
            )

        with NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        extracted_text = extract_text_from_pdf(str(temp_path))

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found. "
                    "Scanned PDFs may require OCR."
                ),
            )

        # Create a unique ID for this uploaded document.
        document_id = str(uuid.uuid4())

        # Split the PDF text into searchable chunks.
        chunks = chunk_document_text(extracted_text)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No searchable text chunks could be created.",
            )

        # Store the chunks and embeddings in ChromaDB.
        chunk_count = index_document_chunks(document_id, chunks)

        return {
            "document_id": document_id,
            "filename": Path(file.filename).name,
            "characters_extracted": len(extracted_text),
            "chunks_indexed": chunk_count,
            "message": (
                "Document uploaded, extracted, and indexed successfully. "
                "Use the document_id with POST /ask-document."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Document indexing error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Document processing failed. Check the server terminal.",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()


@app.post("/ask-document")
def ask_document(request: DocumentQuestion):
    document_id = request.document_id.strip()
    question = request.question.strip()

    if not document_id or not question:
        raise HTTPException(
            status_code=400,
            detail="Both document_id and question are required.",
        )

    try:
        # Search only the selected document's indexed chunks.
        matches = search_document_chunks(
            document_id=document_id,
            query=question,
            limit=4,
        )

        if not matches:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No indexed passages were found for this document. "
                    "Check the document_id and upload the PDF again if needed."
                ),
            )

        context = "\n\n".join(
            f"[Page {match['page']}]\n{match['content']}"
            for match in matches
        )

        answer = generate_answer(question, context)

        return {
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "page": match["page"],
                    "chunk_id": match["id"],
                    "content": match["content"],
                }
                for match in matches
            ],
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"Document question error: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=502,
            detail="Document question failed. Check the server terminal.",
        ) from exc
