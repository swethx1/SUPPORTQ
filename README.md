# SupportQ - AI-Powered Document Support Assistant

SupportIQ is an AI-powered document question-answering system that helps users find information from uploaded documents using Retrieval-Augmented Generation (RAG).

Instead of manually searching through lengthy documents, users can upload a PDF and ask questions in natural language. SupportIQ retrieves relevant text passages and uses a Large Language Model (LLM) to generate contextual answers with supporting source information.

## Features

- **PDF Document Upload:** Upload PDF documents for processing and querying.
- **Retrieval-Augmented Generation (RAG):** Retrieve relevant document passages to ground answers in source material.
- **Semantic Search:** Find relevant content based on meaning rather than relying only on exact keyword matches.
- **AI-Powered Responses:** Generate natural-language answers using the Groq API.
- **Source-Aware Answers:** Display relevant source passages to help users verify responses.
- **FAQ Retrieval:** Search a predefined FAQ knowledge base for relevant information.
- **REST API:** Use FastAPI endpoints to interact with the document processing and question-answering pipeline.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend API | FastAPI |
| LLM Inference | Groq API |
| Vector Database | ChromaDB |
| Text Embeddings | Sentence Transformers |
| PDF Processing | pypdf |
| Frontend Prototype | Streamlit |
| Environment Configuration | python-dotenv |

## How It Works

1. **Document Upload:** The user uploads a PDF through the application.
2. **Text Extraction:** The document processor extracts text from the PDF and preserves page information.
3. **Text Chunking:** Extracted content is divided into smaller, overlapping passages.
4. **Embedding Generation:** Sentence Transformers converts the passages into numerical vector embeddings.
5. **Vector Storage:** ChromaDB stores the document chunks and their embeddings for semantic retrieval.
6. **Question Processing:** The user submits a natural-language question.
7. **Context Retrieval:** SupportIQ retrieves relevant passages associated with the uploaded document.
8. **Answer Generation:** The retrieved context is passed to the LLM through the Groq API to generate an answer.
9. **Source Display:** Relevant passages are returned alongside the response when applicable.

## Architecture

```text
             User
              |
              v
       Streamlit Frontend
              |
              v
         FastAPI Backend
              |
       PDF Text Extraction
              |
              v
        Text Chunking
              |
              v
     Sentence Transformers
       Embedding Generation
              |
              v
           ChromaDB
       Vector Storage
              |
       User's Question
              |
              v
      Semantic Retrieval
              |
              v
      Retrieved Context
              |
              v
          Groq API
              |
              v
       Generated Answer
              |
              v
      Answer and Sources
```

## Getting Started

### Prerequisites

- Python 3.11 or a compatible Python version
- Git
- A Groq API key

### 1. Clone the repository

```bash
git clone https://github.com/swethx1/SUPPORTQ.git
cd SUPPORTQ
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install fastapi uvicorn python-dotenv sentence-transformers chromadb groq python-multipart pypdf streamlit requests
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key. Never commit your `.env` file or expose your API key publicly.

### 5. Run the FastAPI backend

```bash
python -m uvicorn app.main:app --reload
```

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

### 6. Run the Streamlit frontend

Open a second terminal, activate the virtual environment, and run:

```bash
python -m streamlit run frontend.py
```

Open the local URL displayed in the terminal, usually:

http://localhost:8501

## Project Structure

```text
SUPPORTQ/
├── app/
│   ├── __init__.py
│   ├── document_processor.py
│   ├── llm.py
│   ├── main.py
│   └── retriever.py
├── data/
│   └── faqs.json
├── frontend.py
├── .gitignore
└── README.md
```

- `document_processor.py` — Extracts and prepares PDF text.
- `retriever.py` — Handles embeddings, vector storage, and retrieval.
- `llm.py` — Integrates the language model for answer generation.
- `main.py` — Defines the FastAPI application and API endpoints.
- `faqs.json` — Contains the predefined FAQ knowledge base.
- `frontend.py` — Provides the Streamlit user interface.

## API Endpoints

The main application includes endpoints for document upload and document question-answering.

| Endpoint | Purpose |
|---|---|
| `POST /upload-document` | Upload and index a document |
| `POST /ask-document` | Ask a question about an uploaded document |
| `GET /docs` | Explore the API using Swagger UI |

Use the interactive API documentation at `/docs` to inspect request parameters and responses.

## Security Considerations

- Store API keys in environment variables.
- Keep `.env`, virtual environments, local vector databases, and private documents out of version control.
- Do not upload confidential documents to external AI services without appropriate authorisation.
- Validate uploaded files and handle document-processing errors appropriately.

## Future Improvements

- Add web search to supplement document-based answers.
- Improve source citations with page-level references.
- Introduce user authentication and document access controls.
- Add document management features, including deletion and re-indexing.
- Build a polished, responsive custom frontend.
- Add automated tests, logging, and deployment support.

## Learning Outcomes

This project explores practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Semantic search and vector databases
- Text embeddings and document chunking
- LLM API integration
- REST API development using FastAPI
- PDF processing and information retrieval
- Python application development

## Author

**Swetha**

Developed as a hands-on project to explore AI-powered document retrieval, LLM integration, and backend application development.

## License

A license has not yet been selected. Until one is added, the repository should not be assumed to grant permission to reuse or redistribute the code.
