# Traditional RAG

This repository provides a complete, traditional Retrieval-Augmented Generation (RAG) pipeline built with Python. It features a FastAPI backend that answers questions based on a collection of local documents. The system leverages advanced chunking strategies, integrates with Google's Gemini LLM for generation, and uses ChromaDB for efficient vector storage and retrieval.

A key feature of this implementation is the "retrieval with neighbors" strategy, which enriches the context provided to the language model by including chunks adjacent to the most relevant retrieved documents.

## Features
- **FastAPI Backend**: Exposes a simple `/ask` endpoint for querying documents.
- **Flexible Document Loading**: Supports both PDF and TXT file formats out-of-the-box.
- **Advanced Chunking Strategy**: Implements a two-level chunking process:
    1. Documents are first split into logical sections based on headings.
    2. Large sections are then further divided into smaller, overlapping chunks using a `RecursiveCharacterTextSplitter`.
- **State-of-the-Art Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` for generating high-quality text embeddings.
- **Vector Storage**: Utilizes `ChromaDB` for efficient storage and similarity search of document chunks.
- **Context-Enhanced Retrieval**: Implements a `retrieve_with_neighbors` function that fetches not only the most relevant chunks but also their surrounding chunks to provide richer context to the LLM.
- **LLM Integration**: Uses the Google Gemini Pro model for generating answers grounded in the retrieved context.
- **Source Attribution**: Answers include citations pointing to the exact source file and page number from which the information was extracted.
- **Evaluation Suite**: Includes scripts and a dataset (`questions.json`) to evaluate the retrieval performance (Recall@3).

## Architecture
The RAG pipeline is orchestrated through a series of modules in the `src` directory.

1.  **Indexing (`indexer.py`)**:
    -   **Load**: Documents from the `documents/` directory are loaded using `load_all_documents` (`document_loader.py`). PDFs are parsed page by page, and TXT files are read whole.
    -   **Split**: The loaded documents are processed by `split_documents` (`text_splitter.py`). This function first identifies structural sections based on heading patterns (e.g., "1.1 Section Title") and then applies `RecursiveCharacterTextSplitter` to any sections exceeding a defined size. This creates semantically coherent chunks.
    -   **Embed & Store**: Each chunk is converted into a vector embedding using the Hugging Face model (`embeddings.py`). These embeddings, along with their content and metadata, are stored in a persistent Chroma vector store (`vector_store.py`).

2.  **Retrieval and Generation (`main.py`)**:
    -   **API Endpoint**: The FastAPI application exposes an `/ask` endpoint that accepts a user query.
    -   **Retrieve**: The `retrieve_with_neighbors` function (`retriever.py`) is called. It performs a similarity search against the vector store to find the top 3 most relevant chunks (primary documents).
    -   **Expand Context**: For each primary chunk, it identifies and fetches its adjacent chunks (one before and one after) from the complete set of indexed chunks. This creates an expanded, ordered context.
    -   **Generate**: The expanded context, along with the original query, is formatted into a detailed prompt and sent to the Gemini LLM via `generate_answer` (`llm.py`). The prompt explicitly instructs the model to answer only from the provided sources and to include citations.
    -   **Respond**: The generated answer and the source metadata from the primary documents are returned as a JSON response.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/prakhar-singh-17/traditionalrag.git
cd traditionalrag
```

### 2. Install Dependencies
It is recommended to use a virtual environment. This project uses `uv` for dependency management, but `pip` will also work.
```bash
# Using uv
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
You will need an API key for the Google Gemini model.

1.  Create a `.env` file in the root of the project:
    ```bash
    touch .env
    ```
2.  Add your Gemini API key to the file:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

### 4. Add Documents
Place the `.pdf` and `.txt` files you want to query into the `documents/` directory. A sample `yoga.txt` file is included.

## Usage

### 1. Run the Application
Start the FastAPI server using `uvicorn`:
```bash
uvicorn main:app --reload
```
The server will be available at `http://127.0.0.1:8000`.

### 2. Ask a Question
You can interact with the API through your browser, `curl`, or any other API client.

**Browser**:
Navigate to the following URL to ask a question:
`http://127.0.0.1:8000/ask?query=What is yoga?`

**cURL**:
```bash
curl -X GET "http://127.0.0.1:8000/ask?query=What%20is%20yoga%3F"
```

The API will return a JSON object containing the question, the generated answer, and a list of sources.

### 3. API Documentation
Interactive API documentation (provided by Swagger UI) is available at `http://127.0.0.1:8000/docs`.

## Evaluation
The repository includes a script to evaluate the performance of the retrieval component. The evaluation measures Recall@3, which checks if any of the expected document chunks are present in the top 3 retrieved results.

To run the evaluation:
```bash
python evaluation/evaluate_retrieval.py
```
The script uses the questions and expected chunk IDs defined in `evaluation/questions.json`. It will print a summary of the results, including the final recall score.

## Project Structure
```
.
├── documents/          # Source documents (PDFs, TXTs) to be indexed.
│   └── yoga.txt
├── evaluation/         # Scripts and data for evaluating the RAG pipeline.
│   ├── evaluate_retrieval.py
│   ├── inspect_chunks.py
│   └── questions.json
├── src/                # Core source code for the RAG pipeline.
│   ├── document_loader.py    # Loads and parses files from the documents directory.
│   ├── embeddings.py         # Creates the sentence-transformer embedding model.
│   ├── indexer.py            # Orchestrates the document loading, splitting, and indexing process.
│   ├── llm.py                # Handles interaction with the Gemini LLM.
│   ├── retriever.py          # Implements the retrieval strategy with neighbor expansion.
│   ├── text_splitter.py      # Implements the section-based chunking logic.
│   └── vector_store.py       # Manages the ChromaDB vector store.
├── main.py             # FastAPI application entrypoint.
├── requirements.txt    # Project dependencies.
└── ...
