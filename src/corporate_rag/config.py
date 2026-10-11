from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    '''Application settings loaded from environment variables.'''
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )
    
    openai_api_key: str | None = None
    
    max_pdf_size_mb: int = 10
    
    chat_model: str = 'gpt-4.1-mini'
    embedding_model: str = 'text-embedding-3-small'
    
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    
    pdf_dir: Path = PROJECT_ROOT / 'data' / 'pdfs'
    vector_store_dir: Path = PROJECT_ROOT / 'data' / 'chroma'


settings = Settings()
