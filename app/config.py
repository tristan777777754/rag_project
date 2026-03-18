from dotenv import load_dotenv
import os

load_dotenv()

MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
KIMI_BASE_URL = "https://api.moonshot.ai/v1"
CHAT_MODEL = "kimi-k2-0905-preview"

CHROMA_DB_PATH = "data/chroma_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4