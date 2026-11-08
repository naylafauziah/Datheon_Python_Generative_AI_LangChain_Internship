"""Shared project configuration. Every file is resolved from the project root."""
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DOCS_DIR = ROOT / 'data' / 'docs'
SAMPLES_DIR = ROOT / 'data' / 'samples'
OUTPUT_DIR = ROOT / 'outputs'
CHROMA_DIR = ROOT / 'chroma_db'
MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'gemini-embedding-001')
RAG_TOP_K = int(os.getenv('RAG_TOP_K', '4'))


def require_key(name: str) -> str:
    value = os.getenv(name, '').strip()
    if not value or value.startswith('replace_with_'):
        raise RuntimeError(f'{name} belum diatur. Salin .env.example menjadi .env lalu isi API key.')
    return value
