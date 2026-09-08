import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_documents(documents):
    final_chunks = []

    for document in documents:
        document_id = document.metadata.get("source")

        sections = split_into_sections(document)

        for section_index, section in enumerate(sections):
            section_text = section.page_content

            # Keep reasonably sized sections together
            if len(section_text) <= 1000:
                chunks = [section]

            # Split large sections recursively
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_documents([section])

            for chunk_index, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = chunk_index
                chunk.metadata["document_id"] = document_id
                chunk.metadata["section_index"] = section_index
                page_number = chunk.metadata.get("page")
                chunk.metadata["chunk_id"] = (
                    f"{document_id}:{page_number}:{section_index}:{chunk_index}"
                )

            final_chunks.extend(chunks)

    return final_chunks


def split_into_sections(document):
    text = document.page_content

    # Detect headings such as:
    # 6.2 Information security objectives and planning to achieve them
    heading_pattern = r"(?m)^\d+(?:\.\d+)+\s+[^.\n]+$"

    matches = list(re.finditer(heading_pattern, text))

    # If no section headings are found,
    # return the original document unchanged.
    if not matches:
        return [document]

    sections = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        section_heading = match.group().strip()

        sections.append(
            Document(
                page_content=section_text,
                metadata={
                    **document.metadata,
                    "section": section_heading
                }
            )
        )

    return sections