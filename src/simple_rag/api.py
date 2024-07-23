import fastapi
import openai
import pydantic

from . import document_store
from . import qa_engine

router = fastapi.APIRouter(prefix="/v1")


class DocumentInput(pydantic.BaseModel):
    text: str

class QuestionInput(pydantic.BaseModel):
    question: str


@router.post("/documents")
def add_document(document: DocumentInput):
    store = document_store.get_default_document_store()
    try: 
        store.add_documents([document.text])
    except ValueError as e:
        return {"status": "ERROR", "error": str(e)}

    return {"status": "OK"}


@router.get("/documents")
def get_documents():
    store = document_store.get_default_document_store()
    documents = store.get_all_documents()
    return {"status": "OK", "documents": documents}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    store = document_store.get_default_document_store()
    store.delete_document(document_id=document_id)
    return {"status": "OK"}


@router.post("/answer")
def answer_question(question: QuestionInput):
    store = document_store.get_default_document_store()
    qa = qa_engine.QAEngine(openai.OpenAI(), store)
    answer = qa.answer_question(question=question.question)

    if not answer:
        return {"status": "ERROR", "error": "No answer found"}

    answer = {
        "answer": answer[0],
        "documents": answer[1],
    }

    return {"status": "OK", "answers": answer}

