import json

from src.indexer import index_documents
from src.retriever import retrieve_with_neighbors


with open("evaluation/questions.json", "r", encoding="utf-8") as file:
    questions = json.load(file)


vector_store = index_documents("documents")


total_questions = 0
passed_questions = 0


for item in questions:

    query = item["question"]

    expected_chunks = item.get("expected_chunks")
    expected_pages = item.get("expected_pages")

    # ---------------------------------
    # Chunk-level evaluation
    # ---------------------------------

    if expected_chunks:

        total_questions += 1

        retrieved_documents, primary_documents = retrieve_with_neighbors(
            vector_store,
            query,
            neighbor_size=1
        )

        retrieved_chunk_ids = [
            document.metadata.get("chunk_id")
            for document in primary_documents
        ]

        print("\nQuestion:")
        print(query)

        print("\nExpected chunks:")
        print(expected_chunks)

        print("\nRetrieved chunks:")

        for chunk_id in retrieved_chunk_ids:
            print(chunk_id)

        if any(
            chunk_id in expected_chunks
            for chunk_id in retrieved_chunk_ids
        ):
            print("\nResult: PASS")
            passed_questions += 1
        else:
            print("\nResult: FAIL")

        continue

    # ---------------------------------
    # Page-level evaluation
    # ---------------------------------

    if expected_pages:

        total_questions += 1

        retrieved_documents, primary_documents = retrieve_with_neighbors(
            vector_store,
            query,
            neighbor_size=1
        )

        retrieved_pages = [
            document.metadata.get("page")
            for document in primary_documents
        ]

        print("\nQuestion:")
        print(query)

        print("\nExpected pages:")
        print(expected_pages)

        print("\nRetrieved pages:")

        for page in retrieved_pages:
            print(page)

        if any(
            page in expected_pages
            for page in retrieved_pages
        ):
            print("\nResult: PASS")
            passed_questions += 1
        else:
            print("\nResult: FAIL")

        continue

    # ---------------------------------
    # Unanswerable question
    # ---------------------------------

    print("\nQuestion:")
    print(query)

    print("\nEvaluation:")
    print("SKIP - No ground-truth evidence")


print("\n" + "=" * 60)
print("RETRIEVAL EVALUATION SUMMARY")
print("=" * 60)

print(f"Answerable questions: {total_questions}")
print(f"Correctly retrieved: {passed_questions}")

recall_at_3 = passed_questions / total_questions

print(f"Recall@3: {recall_at_3:.2%}")