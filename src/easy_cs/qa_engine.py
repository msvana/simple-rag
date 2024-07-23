import json

import openai

from . import document_store

PROMPT_TEMPLATE = """
You are an expert on answering questions related to the following documents:

----- DOCUMENT LIST START -----
{documents}
----- DOCUMENT LIST END -----

The user asked you the following question: 

----- QUESTION START -----
{question}
----- QUESTION END -----

Please provide an answer to the user's question. If the answer cannot be found in the documents above
answer with as a special token DONT_KNOW to indicate that you do not know the answer.
"""

# A function call ensures a deterministic answer structure
FUNCTION_DEFINITION = {
    "name": "answer_question",
    "description": "Returns the answer to a question based on a set of documents.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question.",
            },
        },
    },
}


class QAEngine:

    def __init__(
        self, openai_client: openai.OpenAI, document_store: document_store.DocumentStore
    ):
        self._openai_client = openai_client
        self._document_store = document_store

    def answer_question(
        self, question: str, n_documents: int = 5
    ) -> tuple[str, list[str]] | None:
        documents = self._document_store.query(question, n_results=n_documents)

        if not documents:
            return None

        documents_string = "\n----- DOCUMENT SEPARATOR -----\n".join(documents)
        prompt = PROMPT_TEMPLATE.format(documents=documents_string, question=question)

        response = self._openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
            tools=[
                {"type": "function", "function": FUNCTION_DEFINITION}  # type: ignore
            ],
            tool_choice={"type": "function", "function": {"name": "answer_question"}},
        )

        tool_calls = response.choices[0].message.tool_calls

        if not tool_calls:
            return None

        args_str = tool_calls[0].function.arguments
        args = json.loads(args_str)

        if args["answer"] == "DONT_KNOW":
            return None

        return args["answer"], documents
