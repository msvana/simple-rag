import chromadb
import pytest


@pytest.fixture
def chroma_collection():
    client = chromadb.Client()
    collection = client.create_collection("test_collection")
    yield collection
    client.delete_collection("test_collection")
