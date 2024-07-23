import chromadb
import openai

from base import chroma_collection
from easy_cs import qa_engine
from easy_cs import document_store


EXAMPLE_DOCUMENTS = [
    # A document about Python
    """
    Python is an interpreted high-level general-purpose programming language. 
    Python's design philosophy emphasizes code readability with its notable use of significant indentation. 
    Its language constructs as well as its object-oriented approach aim to help programmers write clear, 
    logical code for small and large-scale projects.

    PEP 8 is a style guide that provides a set of rules and best practices for writing Python code.
    """,
    # A document about JavaScript
    """
    JavaScript is a high-level programming language that conforms to the ECMAScript specification.
    JavaScript is a multi-paradigm language, supporting object-oriented, imperative, and declarative
    programming styles.

    JavaScript is the most popular programming language in the world, and is used to create interactive
    websites.

    React is a popular JavaScript library for building user interfaces. It is maintained by Facebook.
    """,
]


def test_qeuestion_answering(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=EXAMPLE_DOCUMENTS)
    openai_client = openai.OpenAI()

    engine = qa_engine.QAEngine(openai_client=openai_client, document_store=store)
    response = engine.answer_question(
        question="Does Python support object-oriented programming?", n_documents=1
    )

    assert response is not None

    answer, documents = response

    assert answer != "DONT_KNOW"
    assert len(documents) == 1
    assert documents[0] == EXAMPLE_DOCUMENTS[0]


def test_question_answering_no_documents(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    openai_client = openai.OpenAI()

    engine = qa_engine.QAEngine(openai_client=openai_client, document_store=store)
    response = engine.answer_question(
        question="Does Python support object-oriented programming?", n_documents=1
    )

    assert response is None


def test_question_answering_no_answer(chroma_collection: chromadb.Collection):
    store = document_store.DocumentStore(collection=chroma_collection)
    store.add_documents(documents=EXAMPLE_DOCUMENTS)
    openai_client = openai.OpenAI()

    engine = qa_engine.QAEngine(openai_client=openai_client, document_store=store)
    response = engine.answer_question(
        question="What is the capital of France?", n_documents=1
    )

    assert response is None
