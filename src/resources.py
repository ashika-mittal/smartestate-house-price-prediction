from functools import lru_cache
import json
import os
import pickle

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from src.config import (
    COLUMNS_PATH,
    DATA_PATH,
    EMBEDDING_MODEL_NAME,
    ENV_PATH,
    MODEL_PATH
)


load_dotenv(ENV_PATH)


@lru_cache(maxsize=1)
def load_model():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(COLUMNS_PATH, "r") as columns_file:
        model_columns = json.load(columns_file)

    return model, model_columns


@lru_cache(maxsize=1)
def load_reference_data():
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)