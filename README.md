# Simple RAG

This is an implementation of a simple RAG (Retrieval-Augmented Generation)
system allowing you to create a knowledge base by adding text documents and
then querying it with natural language questions.

It is build on top of (FastAPI)[https://fastapi.tiangolo.com/], 
(Chroma)[https://trychroma.com/], and (OpenAI's GPT models)[https://openai.com/].

## Installation

In this project I used Rye to manage the Python environment and dependencies.
You can install Rye by following the instructions at (their website)
[https://rye.astral.sh/].

After installing Rye, clone this and create the environment by running:

```bash
rye sync
```

## Running the API server

First start the Chroma server by running (uses port 8000):

```bash
rye run db
```

Before starting the FastAPI server, you need to set the `OPENAI_API_KEY` environment variable:

```bash
export OPEN_AI_API_KEY=your_openai_api_key
```

Then start the FastAPI server by running:

```bash
rye run api
```

The API server will be available at `http://localhost:5000`.

If you want more configuration (e.g. changing the port), you can also run the
`fastapi` command directly, for example:

```bash
rye run fastapi run src/simple_rag/main.py --port 6000
```

## Basic usage

### Adding documents

You can add documents to the knowledge base by sending a POST request to
`/v1/documents` with a JSON body containing the `text` of the document:

```bash
curl -X POST "http://localhost:5000/v1/documents" -H "Content-Type: application/json" -d '{"text": "The capital of France is Paris."}'
```

### Querying the knowledge base

You can query the knowledge base by sending a POST request to `/v1/answer` with
a JSON body containing the `question` you want to ask:

```bash
curl -X POST "http://localhost:5000/v1/answer" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
```

The prompt used to generate the answer prevents the model from answering the question if the answer
is not in the knowledge base. In other words, GPT won't use its general knowledge to answer the question.

If your question cannot be answered, an error message will be returned.

### Listing documents

You can list all documents in the knowledge base by sending a GET request to
`/v1/documents`:

```bash
curl -X GET "http://localhost:5000/v1/documents"
```

### Deleting documents

You can delete documents from the knowledge base by sending a DELETE request to
`/v1/documents` with a JSON body containing the `text` of the document:

```bash
curl -X DELETE "http://localhost:5000/v1/documents/{documentId}"
```

## Configuration

You can configure the behavior of the RAG system by changing the values in
`src/simple_rag/config.py`. The available configuration options are.

## Future improvements

- Improve documentation - especially output descriptions and error messages
- Make changing ports and hosts for chroma and fastapi easier
- Create a docker-compose file to run the whole system with a single command
- Check for document duplicates
- Control for the total length of the retrieved documents, so we don't exceed the size of the context window
- Add document splitting to allow for larger documents
