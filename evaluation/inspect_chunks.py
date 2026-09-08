from src.indexer import index_documents


vector_store = index_documents("documents")

data = vector_store.get()

documents = data["documents"]
metadatas = data["metadatas"]


for document, metadata in zip(documents, metadatas):

    if metadata.get("page") == 5:

        print("\n" + "=" * 70)

        print(
            f"Chunk ID: {metadata.get('chunk_id')}"
        )

        print(
            f"Page: {metadata.get('page')}"
        )

        print(
            f"Section: {metadata.get('section')}"
        )

        print("-" * 70)

        print(document)