import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = os.getenv("CHROMA_DB_PERSIST_DIR", "./chroma_db")