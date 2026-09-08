from fastapi import FastAPI

from src.indexer import index_documents
from src.retriever import retrieve_with_neighbors
from src.llm import create_gemini_client, generate_answer


app = FastAPI()


# -----------------------------
# Initialize RAG components
# -----------------------------

vector_store = index_documents("documents")

gemini_client = create_gemini_client()


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "Traditional RAG backend is working!"
    }


@app.get("/ask")
def ask(query: str):

    # Retrieve relevant chunks + neighboring chunks
    retrieved_documents, primary_documents = retrieve_with_neighbors(
        vector_store,
        query,
        neighbor_size=1
    )

    print("\n" + "=" * 60)
    print("FINAL CONTEXT SENT TO GEMINI")
    print("=" * 60)

    for document in retrieved_documents:
        print(
            f"\nChunk ID: {document.metadata.get('chunk_id')}"
        )
        print(
            f"Page: {document.metadata.get('page', 'N/A')}"
        )
        print(
            f"Section: {document.metadata.get('section', 'N/A')}"
        )
        print("-" * 60)
        print(document.page_content)
    # Generate answer using Gemini
    answer = generate_answer(
        gemini_client,
        query,
        retrieved_documents
    )

    # Prepare source information
    sources = []

    for document in primary_documents:
        sources.append({
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page"),
            "chunk_id": document.metadata.get("chunk_id"),
            "file_type": document.metadata.get("file_type")
        })

    return {
        "question": query,
        "answer": answer,
        "sources": sources
    }