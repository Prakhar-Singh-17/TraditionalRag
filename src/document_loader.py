import pymupdf
from pathlib import Path
from langchain_core.documents import Document


def load_pdf(file_path):
    pdf = pymupdf.open(file_path)

    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "page": page_number,
                    "file_type": "pdf",
                },
            )
        )

    pdf.close()

    return documents


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return Document(
        page_content=text, metadata={"source": file_path.name, "file_type": "txt"}
    )


def load_all_documents(folder_path):
    documents = []

    folder = Path(folder_path)

    for file_path in folder.iterdir():

        if file_path.suffix.lower() == ".txt":
            documents.append(load_txt(file_path))

        if file_path.suffix.lower() == ".pdf":
            documents.extend(load_pdf(file_path))

    return documents
