from langchain_chroma import Chroma


def create_vector_store(embeddings, chunks):
    vector_store = Chroma(
        collection_name="traditional_rag",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    vector_store.add_documents(chunks)

    return vector_store