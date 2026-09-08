from .document_loader import load_all_documents
from .text_splitter import split_documents
from .embeddings import create_embedding_model
from .vector_store import create_vector_store


def index_documents(folder_path):
    # 1. Load PDF/TXT files
    documents = load_all_documents(folder_path)

    # 2. Split documents into chunks
    chunks = split_documents(documents)

    # 3. Create embedding model
    embeddings = create_embedding_model()

    # 4. Create and populate vector store
    vector_store = create_vector_store(embeddings,chunks)

    return vector_store