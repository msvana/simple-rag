import os

BASE_PATH = os.path.dirname(os.path.dirname(__file__))

CHROMA_COLLECTION = "documents"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 0.05
OPENAI_MAX_TOKENS = 1024

NUM_DOCUMENTS_RETRIEVED = 5
