import os

from fastapi import testclient

from simple_rag import main
from simple_rag import document_store
from simple_rag import config


main.app.dependency_overrides[document_store.get_document_store] = (
    document_store.get_document_store_for_testing
)

test_client = testclient.TestClient(main.app)


def test_add_document():
    response = test_client.post(
        "/v1/documents",
        json={"text": "This is a test document"},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

    store = document_store.get_document_store_for_testing()
    documents = store.get_all_documents()
    assert len(documents) == 1
    assert documents[0][1] == "This is a test document"

    store._collection._client.delete_collection(config.CHROMA_COLLECTION)


def test_add_document_empty():
    response = test_client.post(
        "/v1/documents", json={"text": ""}, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["status"] == "ERROR"


def test_get_documents():
    store = document_store.get_document_store_for_testing()
    store.add_documents(["Document 1", "Document 2"])

    response = test_client.get("/v1/documents")
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["status"] == "OK"
    assert len(response_json["documents"]) == 2
    assert response_json["documents"][0][1] in ["Document 1", "Document 2"]
    assert response_json["documents"][1][1] in ["Document 1", "Document 2"]
    assert response_json["documents"][0][1] != response_json["documents"][1][1]

    assert isinstance(response_json["documents"][0][0], str)
    assert isinstance(response_json["documents"][0][1], str)

    store._collection._client.delete_collection(config.CHROMA_COLLECTION)


def test_get_documents_empty():
    response = test_client.get("/v1/documents")
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["status"] == "OK"
    assert len(response_json["documents"]) == 0


def test_delete_document():
    store = document_store.get_document_store_for_testing()
    store.add_documents(["Document 1", "Document 2"])

    documents = store.get_all_documents()
    document_id = [d[0] for d in documents if d[1] == "Document 1"][0]

    response = test_client.delete(f"/v1/documents/{document_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

    documents = store.get_all_documents()
    assert len(documents) == 1
    assert documents[0][1] == "Document 2"

    store._collection._client.delete_collection(config.CHROMA_COLLECTION)


def test_delete_document_non_existent():
    store = document_store.get_document_store_for_testing()
    store.add_documents(["Document 1", "Document 2"])
    response = test_client.delete("/v1/documents/some-random-id")

    assert response.status_code == 200
    assert response.json()["status"] == "OK"

    assert len(store.get_all_documents()) == 2


def test_answer_question():
    tests_path = os.path.dirname(os.path.abspath(__file__))
    test_file_path = os.path.join(tests_path, "data", "no_sale_countries.md")

    with open(test_file_path, "r") as f:
        test_document = f.read()

    test_client.post(
        "/v1/documents",
        json={"text": test_document},
        headers={"Content-Type": "application/json"},
    )

    response = test_client.post(
        "/v1/answer",
        json={"question": "Which countries are not part of the sale?"},
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "OK"

    answer = response_json["answer"]
    assert "spain" in answer.lower()
    assert "italy" in answer.lower()
    assert "germany" in answer.lower()
    assert "sweden" in answer.lower()

    store = document_store.get_document_store_for_testing()
    store._collection._client.delete_collection(config.CHROMA_COLLECTION)

def test_answer_no_documents():
    response = test_client.post(
        "/v1/answer",
        json={"question": "Which countries are not part of the sale?"},
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 404
    response_json = response.json()
    assert response_json["status"] == "ERROR"
    assert response_json["error"] == "NO_ANSWER"


def test_answer_no_answer():
    test_client.post(
        "/v1/documents",
        json={"text": "Pyhon is a programming language."},
        headers={"Content-Type": "application/json"},
    )

    response = test_client.post(
        "/v1/answer",
        json={"question": "What is the capital of France?"},
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 404
    response_json = response.json()
    assert response_json["status"] == "ERROR"
    assert response_json["error"] == "NO_ANSWER"

    store = document_store.get_document_store_for_testing()
    store._collection._client.delete_collection(config.CHROMA_COLLECTION)
