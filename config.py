from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
TOP_K=5
CHROMA_PATH=str(ROOT/'chroma_db')
COLLECTION='per04_service'
MODEL='GigaChat-2-Pro'
ENV_FILE=os.getenv('RAG_PROVIDER_ENV',str(ROOT / '.env'))
os.environ.setdefault('FASTEMBED_CACHE_PATH',str(ROOT / 'model-cache'))


