import fastapi
import openai
import pydantic

from . import document_store
from . import qa_engine
from . import config

router = fastapi.APIRouter(prefix="/v1")


class DocumentInput(pydantic.BaseModel):
    text: str


class QuestionInput(pydantic.BaseModel):
    question: str


@router.post("/documents")
def add_document(
    document: DocumentInput,
    response: fastapi.Response,
    store: document_store.DocumentStore = fastapi.Depends(
        document_store.get_document_store
    ),
):
    """
    Add a new document to the document store

    Example:

    ```
    curl -X 'POST' \\
        'http://localhost:5000/v1/documents' \\
        -H 'accept: application/json' \\
        -H 'Content-Type: application/json' \\
        -d '{ "text": "The capital of France is Paris" }'
    ```
    """

    try:
        store.add_documents([document.text])
    except ValueError as e:
        response.status_code = fastapi.status.HTTP_400_BAD_REQUEST
        return {"status": "ERROR", "error": str(e)}

    return {"status": "OK"}


@router.get("/documents")
def get_documents(
    store: document_store.DocumentStore = fastapi.Depends(
        document_store.get_document_store
    ),
):
    """
    Get all documents from the document store. 
    Returns a list of document IDs and texts.

    Example:
    ```
    curl -X 'GET' \\
        'http://localhost:5000/v1/documents' \\
        -H 'accept: application/json'
    ```
    """
    documents = store.get_all_documents()
    return {"status": "OK", "documents": documents}


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    store: document_store.DocumentStore = fastapi.Depends(
        document_store.get_document_store
    ),
):
    """
    Delete a document from the document store

    Example:
    ```
    curl -X 'DELETE' \\
        'http://localhost:5000/v1/documents/2739349c-ae07-479b-8843-d16a5ece4099' \\
        -H 'accept: application/json'
    ```
    """
    store.delete_document(document_id=document_id)
    return {"status": "OK"}


@router.post("/answer")
def answer_question(
    question: QuestionInput,
    response: fastapi.Response,
    store: document_store.DocumentStore = fastapi.Depends(
        document_store.get_document_store
    ),
):
    """
    Answer a question using documents stored in the document store.
    Returns the answer and the documents used to generate the answer.

    Example:
    ```
    curl -X 'POST' \\
        'http://localhost:5000/v1/answer' \\
        -H 'accept: application/json' \\
        -H 'Content-Type: application/json' \\
        -d '{ "question": "What is the capital of France?" }'
    ```

    """

    qa = qa_engine.QAEngine(openai.OpenAI(api_key=config.OPENAI_API_KEY), store)
    answer = qa.answer_question(
        question=question.question, n_documents=config.NUM_DOCUMENTS_RETRIEVED
    )

    if not answer:
        response.status_code = fastapi.status.HTTP_404_NOT_FOUND
        return {"status": "ERROR", "error": "NO_ANSWER"}

    return {"status": "OK", "answer": answer[0], "documents": answer[1]}
