# EngiDoc AI — Engineering Documentation Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions about engineering documents using semantic search, cross-encoder reranking, and source-grounded LLM responses.

## Features

- **PDF ingestion:** Extract text from engineering manuals while preserving page-level metadata.
- **Text chunking:** Split documents into overlapping chunks for retrieval.
- **Semantic search:** Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector storage:** Persist and search document embeddings using ChromaDB.
- **Cross-encoder reranking:** Reorder retrieved passages by query relevance.
- **Grounded answers:** Generate responses using Groq-hosted LLMs and retrieved document evidence.
- **Source citations:** Connect answers to supporting document passages and page numbers.
- **Abstention behavior:** Acknowledge insufficient evidence rather than intentionally inventing unsupported specifications.
- **Web interface and API:** Use the Streamlit frontend or FastAPI endpoints.
- **Containerization:** Run the frontend and backend with Docker Compose.
- **Automated testing:** Test ingestion, chunking, retrieval, reranking, and question-answering components.

## Architecture

```text
Engineering PDF
      |
      v
PDF Text Extraction
      |
      v
Page-Aware Chunking
      |
      v
Embedding Model
      |
      v
ChromaDB Vector Store
      |
      v
User Question
      |
      v
Semantic Retrieval
      |
      v
Cross-Encoder Reranking
      |
      v
LLM Answer Generation
      |
      v
Answer + Source Citations
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF processing | pypdf |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| Reranking | Cross-Encoder |
| LLM integration | Groq |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Testing | pytest |

## Project Structure

```text
engidoc-ai/
├── app/
│   ├── ingestion/       # PDF loading and chunking
│   ├── retrieval/       # Embeddings, vector store, reranking
│   ├── rag/             # Answer generation and QA orchestration
│   ├── api.py           # FastAPI application
│   └── frontend.py      # Streamlit interface
├── sample_docs/         # Sample engineering documents
├── tests/               # Automated tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .gitignore
```

## Getting Started

### Prerequisites

- Python and pip for local development
- Docker Desktop and Docker Compose for containerized execution
- A Groq API key
- An indexed ChromaDB collection populated with engineering documents

### Configure environment variables

Create a local `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit `.env` or expose API keys in source code, logs, screenshots, or public repositories.

### Run with Docker Compose

After configuring your environment and preparing the local ChromaDB index at `data/chroma`, run:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Open the application:

- **Streamlit UI:** http://localhost:8501
- **FastAPI:** http://localhost:8000
- **Interactive API documentation:** http://localhost:8000/docs
- **Health endpoint:** http://localhost:8000/health

View logs:

```bash
docker compose logs --tail=100 api
docker compose logs --tail=100 frontend
```

Stop the application:

```bash
docker compose down
```

The ChromaDB directory is stored locally and mounted into the API container. It is intentionally not included in Git.

> **Note:** A fresh clone does not contain the local vector database. Populate and index your documents before starting the API. Use the ingestion and indexing workflow in the source code to prepare the collection.

## Run Tests

Create and activate a virtual environment, install the dependencies, and run:

```bash
python -m pytest -v
```

The current development baseline passed **32 automated tests**. Run the suite again after making changes to verify the current code.

## Retrieval Evaluation

A small development evaluation compared retrieval with and without cross-encoder reranking.

| Metric | Baseline | Reranked |
|---|---:|---:|
| Precision@5 | 0.533 | 0.667 |
| Page Recall@5 | 1.000 | 1.000 |
| MRR@5 | 0.750 | 1.000 |
| Abstention | 100% | 100% |

In this evaluation, reranking improved Precision@5 by approximately 13.4 percentage points and MRR@5 from 0.75 to 1.00.

These results come from a small evaluation set; expected-page labels are only a proxy for relevance. They should not be interpreted as a general performance guarantee. The abstention result also reflects the tested questions, not a comprehensive measure of hallucination resistance.

## API Usage

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the engineering manual say about the component?"
  }'
```

Replace the example question with a question relevant to your indexed documents. Inspect `/docs` for the exact request schema and response fields.

## Design Considerations

- Retrieved document content is treated as evidence, not as trusted instructions.
- Answers should cite the supporting retrieved sources.
- Missing evidence should result in an explicit limitation rather than an invented specification.
- Engineering specifications, units, and safety-related statements require verification against the original document.
- Evaluation metrics should be reported with their dataset size and limitations.

## Future Improvements

- Hybrid retrieval combining semantic and lexical search
- Expanded retrieval evaluation with a larger, manually reviewed question set
- CI workflow to run tests on each pull request
- Document upload and re-indexing interface
- Request tracing, latency measurements, and retrieval observability
- Improved deployment configuration and operational monitoring

