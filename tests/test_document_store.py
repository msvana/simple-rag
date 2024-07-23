import chromadb
import pytest

from base import chroma_collection
from easy_cs import document_store


def test_add_documents(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=["hello world", "how are you doing today"])
    result = chroma_collection.get()

    if not result or result["documents"] is None:
        assert False

    assert len(result["documents"]) == 2
    assert result["documents"].index("hello world") != -1
    assert result["documents"].index("how are you doing today") != -1


def test_add_no_documents(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)

    with pytest.raises(ValueError):
        store.add_documents(documents=[])


def test_query(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=["hello world", "this is a rag test", "hi"])
    documents = store.query(query="hello", n_results=2)

    assert len(documents) == 2
    assert documents.index("hello world") != -1
    assert documents.index("hi") != -1


def test_get_all_documents(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=["hello world", "this is a rag test", "hi"])
    documents = store.get_all_documents()

    assert len(documents) == 3

    for doc in documents:
        assert isinstance(doc[0], str)
        assert doc[1] in ["hello world", "this is a rag test", "hi"]


def test_get_all_documents_empty(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    documents = store.get_all_documents()
    assert len(documents) == 0

def test_delete_document(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=["hello world", "this is a rag test", "hi"])
    documents = store.get_all_documents()
    store.delete_document(document_id=documents[0][0])
    documents = store.get_all_documents()
    assert len(documents) == 2

def test_delete_document_non_existent(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=["hello world", "this is a rag test", "hi"])
    documents = store.get_all_documents()
    store.delete_document(document_id="non_existent")
    documents = store.get_all_documents()
    assert len(documents) == 3
