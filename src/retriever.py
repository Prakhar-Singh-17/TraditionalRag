from langchain_core.documents import Document
from langchain_chroma import Chroma


def retrieve_with_neighbors(
    vector_store: Chroma,
    query: str,
    neighbor_size: int = 1
) -> tuple[list[Document], list[Document]]:

    # 1. Find the most relevant chunks along with their scores
    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    # 2. Keep only the Documents as primary results
    primary_documents = [
        document
        for document, score in results
    ]


    # 4. Get all stored chunks from Chroma
    all_chunks = vector_store.get()

    documents = all_chunks["documents"]
    metadatas = all_chunks["metadatas"]

    # 5. Create a lookup using document + section + chunk index
    chunks_by_position = {}

    for document, metadata in zip(documents, metadatas):

        document_id = metadata["document_id"]
        page = metadata.get("page")
        section_index = metadata["section_index"]
        chunk_index = metadata["chunk_index"]

        position = (
            document_id,
            page,
            section_index,
            chunk_index
        )

        chunks_by_position[position] = Document(
            page_content=document,
            metadata=metadata
        )

    # 6. Expand each relevant chunk with its neighbors
    selected_chunks = {}

    for result in primary_documents:

        document_id = result.metadata["document_id"]
        page = result.metadata.get("page")
        section_index = result.metadata["section_index"]
        chunk_index = result.metadata["chunk_index"]

        for offset in range(
            -neighbor_size,
            neighbor_size + 1
        ):

            neighbor_position = (
                document_id,
                page,
                section_index,
                chunk_index + offset
            )

            if neighbor_position in chunks_by_position:
                selected_chunks[neighbor_position] = (
                    chunks_by_position[neighbor_position]
                )

    # 7. Sort chunks by document, section, and chunk order
    selected_chunks = dict(
        sorted(selected_chunks.items())
    )

    # 8. Convert back to a list of Documents
    retrieved_documents = list(
        selected_chunks.values()
    )

    return retrieved_documents, primary_documents