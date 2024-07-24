import uuid

import chromadb

from . import config


class DocumentStore:

    def __init__(self, collection: chromadb.Collection):
        self._collection = collection

    def add_documents(self, documents: list[str]):
        if not documents:
            raise ValueError("No documents provided")

        for document in documents:
            if not document:
                raise ValueError("Document cannot be empty")

        document_ids = [str(uuid.uuid4()) for _ in documents]
        self._collection.add(documents=documents, ids=document_ids)

    def query(self, query: str, n_results: int = 1) -> list[str]:
        results = self._collection.query(query_texts=[query], n_results=n_results)

        if not results:
            return []

        documents = results["documents"]

        if not documents:
            return []

        return documents[0]

    def get_all_documents(self) -> list[tuple[str, str]]:
        result = self._collection.get()

        if not result or result["documents"] is None:
            return []

        return list(zip(result["ids"], result["documents"]))

    def delete_document(self, document_id: str):
        self._collection.delete(ids=[document_id])

        
def get_document_store() -> DocumentStore:
    client = chromadb.PersistentClient() 
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    return DocumentStore(collection=collection)


def get_document_store_for_testing() -> DocumentStore:
    client = chromadb.Client()
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    return DocumentStore(collection=collection)
