from google import genai
from dotenv import load_dotenv
import os

load_dotenv()


def create_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

    return client

def generate_answer(client, query, retrieved_documents):
    context_parts = []

    for document in retrieved_documents:

        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "N/A")
        chunk_id = document.metadata.get("chunk_id", "Unknown")

        context_parts.append(
            f"""
    [Source: {source}, Page: {page}]
    Chunk ID: {chunk_id}

    {document.page_content}
    """
        )
    
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information provided in the sources below.

Important rules:

1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not supported by the sources, say:
   "I could not find the answer in the provided documents."
4. Cite the source after each factual statement using the exact format:
   [Source: filename, Page: number]
5. Only cite sources that actually support the statement.
6. Do not invent source names or page numbers.

Sources:

{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text