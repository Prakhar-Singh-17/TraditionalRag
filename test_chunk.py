from src.document_loader import load_all_documents
from src.text_splitter import split_documents, split_into_sections

docs = load_all_documents("documents")

for document in docs:

    if document.metadata.get("source") != "sample.pdf":
        continue

    if document.metadata.get("page") != 8:
        continue

    sections = split_into_sections(document)

    print("Sections found:", len(sections))
    print()

    for section in sections:

        print("Section:", repr(section.metadata.get("section")))
        print("Characters:", len(section.page_content))
        print()

        # Test any section larger than 1000 characters
        if len(section.page_content) > 1000:

            print(">>> Testing this large section")

            all_chunks = split_documents([document])

            matching_chunks = [
                chunk
                for chunk in all_chunks
                if chunk.metadata.get("section") == section.metadata.get("section")
            ]

            print("Chunks produced:", len(matching_chunks))
            print()

for index, chunk in enumerate(matching_chunks, start=1):
    print(f"Chunk {index}")
    print("Characters:", len(chunk.page_content))
    print("Chunk index:", chunk.metadata.get("chunk_index"))
    print("Section:", repr(chunk.metadata.get("section")))
    print("Page:", chunk.metadata.get("page"))
    print("Source:", chunk.metadata.get("source"))
    print("Chunk ID:", chunk.metadata.get("chunk_id"))
    print("-" * 50)
