from pathlib import Path
from corporate_rag.config import Settings


def test_default_settings():
    config = Settings(_env_file=None)

    assert config.chat_model == "gpt-4.1-mini"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.retrieval_k == 4


def test_api_key_can_be_missing():
    config = Settings(_env_file=None)

    assert config.openai_api_key is None


def test_paths_are_defined():
    config = Settings(_env_file=None)

    assert config.pdf_dir == Path("data/pdfs") or config.pdf_dir.is_absolute()
    assert config.vector_store_dir.name == "chroma"


def test_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "test-model")
    monkeypatch.setenv("RETRIEVAL_K", "6")

    config = Settings(_env_file=None)

    assert config.chat_model == "test-model"
    assert config.retrieval_k == 6
    