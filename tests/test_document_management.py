
from unittest.mock import Mock

import pytest

from corporate_rag.document_management import (
    clear_vector_store,
    delete_document,
)
from corporate_rag.ingestion import load_pdf


def test_load_pdf_rejects_invalid_signature(tmp_path):
    pdf_path = tmp_path / 'invalid.pdf'
    pdf_path.write_text('This is not a PDF.', encoding='utf-8')

    with pytest.raises(ValueError, match='valid PDF signature'):
        load_pdf(pdf_path)


def test_load_pdf_rejects_oversized_file(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / 'large.pdf'
    pdf_path.write_bytes(b'%PDF-' + b'x' * 100)

    monkeypatch.setattr(
        'corporate_rag.ingestion.settings.max_pdf_size_mb',
        0,
    )

    with pytest.raises(ValueError, match='maximum size'):
        load_pdf(pdf_path)


def test_delete_document_removes_matching_chunks():
    vector_store = Mock()
    vector_store.get.return_value = {
        'ids': ['chunk-1', 'chunk-2'],
    }

    count = delete_document('handbook.pdf', vector_store)

    assert count == 2
    vector_store.get.assert_called_once_with(
        where={'source': 'handbook.pdf'},
    )
    vector_store.delete.assert_called_once_with(
        ids=['chunk-1', 'chunk-2'],
    )


def test_delete_document_rejects_empty_source():
    vector_store = Mock()

    with pytest.raises(ValueError, match='cannot be empty'):
        delete_document('   ', vector_store)

    vector_store.get.assert_not_called()


def test_clear_vector_store_removes_all_chunks():
    vector_store = Mock()
    vector_store.get.return_value = {
        'ids': ['chunk-1', 'chunk-2', 'chunk-3'],
    }

    count = clear_vector_store(vector_store)

    assert count == 3
    vector_store.delete.assert_called_once_with(
        ids=['chunk-1', 'chunk-2', 'chunk-3'],
    )


def test_clear_empty_vector_store():
    vector_store = Mock()
    vector_store.get.return_value = {'ids': []}

    count = clear_vector_store(vector_store)

    assert count == 0
    vector_store.delete.assert_not_called()
    