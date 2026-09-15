from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "Models" / "house_price_model.pkl"
COLUMNS_PATH = BASE_DIR / "Models" / "model_columns.json"
DATA_PATH = BASE_DIR / "Data" / "processed" / "clean_house_data.csv"
ENV_PATH = BASE_DIR / ".env"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Groq retired llama-3.1-8b-instant. GPT-OSS 20B is its recommended
# production replacement and is available through the same chat API.
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

DEFAULT_TOP_K = 5
