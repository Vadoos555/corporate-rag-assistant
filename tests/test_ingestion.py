from pathlib import Path

import pytest

from langchain_core.documents import Document
from langchain_core.documents import Document

from corporate_rag.ingestion import load_pdf
import corporate_rag.ingestion as ingestion


def test_missing_pdf_raises_error():
    path = Path("data/pdfs/missing.pdf")

    with pytest.raises(FileNotFoundError):
        load_pdf(path)


def test_non_pdf_file_raises_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello world", encoding="utf-8")

    with pytest.raises(ValueError):
        load_pdf(path)


def test_directory_raises_error(tmp_path):
    with pytest.raises(ValueError):
        load_pdf(tmp_path)


def test_load_pdfs_from_directory(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()

    (pdf_dir / "handbook.pdf").touch()
    (pdf_dir / "security.pdf").touch()

    def fake_load_pdf(file_path: Path) -> list[Document]:
        return [
            Document(
                page_content=f"Content from {file_path.name}",
                metadata={"source": str(file_path), "page": 0},
            )
        ]

    monkeypatch.setattr(ingestion, "load_pdf", fake_load_pdf)

    documents = ingestion.load_pdfs(pdf_dir)

    assert len(documents) == 2
    assert "handbook.pdf" in documents[0].page_content
    assert "security.pdf" in documents[1].page_content


def test_load_pdfs_from_empty_directory(tmp_path):
    documents = ingestion.load_pdfs(tmp_path)

    assert documents == []


def test_missing_pdf_directory_raises_error(tmp_path):
    pdf_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ingestion.load_pdfs(pdf_dir)


def test_validate_documents_removes_empty_text():
    documents = [
        Document(page_content="Employee handbook", metadata={"page": 0}),
        Document(page_content="", metadata={"page": 1}),
        Document(page_content="   \n  ", metadata={"page": 2}),
        Document(page_content="Security policy", metadata={"page": 3}),
    ]

    valid_documents = ingestion.validate_documents(documents)

    assert len(valid_documents) == 2
    assert valid_documents[0].page_content == "Employee handbook"
    assert valid_documents[1].page_content == "Security policy"


def test_validate_documents_returns_empty_list():
    documents = [
        Document(page_content="", metadata={"page": 0}),
        Document(page_content="   ", metadata={"page": 1}),
    ]

    assert ingestion.validate_documents(documents) == []
    