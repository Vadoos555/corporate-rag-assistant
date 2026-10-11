
from unittest.mock import patch

from langchain_core.documents import Document

from corporate_rag.embeddings import create_embeddings
from corporate_rag.indexing import index_documents
from corporate_rag.retrieval import search_documents
from corporate_rag.vector_store import create_vector_store


def test_create_embeddings():
    with (
        patch('corporate_rag.embeddings.settings') as mock_settings,
        patch(
            'corporate_rag.embeddings.OpenAIEmbeddings'
        ) as mock_embeddings,
    ):
        mock_settings.embedding_model = 'text-embedding-3-small'
        mock_settings.openai_api_key = 'test-key'

        create_embeddings()

    mock_embeddings.assert_called_once_with(
        model='text-embedding-3-small',
        api_key='test-key',
    )


def test_create_vector_store(tmp_path):
    with patch(
        'corporate_rag.vector_store.Chroma'
    ) as mock_chroma:
        create_vector_store(
            embeddings=object(),
            persist_directory=tmp_path / 'vectors',
        )

    mock_chroma.assert_called_once()
    assert (tmp_path / 'vectors').is_dir()


def test_index_documents():
    documents = [
        Document(
            page_content='Remote work policy',
            metadata={
                'source': 'handbook.pdf',
                'page': 0,
                'start_index': 0,
            },
        ),
    ]

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids

    vector_store = FakeVectorStore()
    ids = index_documents(documents, vector_store)

    assert len(ids) == 1
    assert len(ids[0]) == 64
    assert vector_store.documents == documents
    assert vector_store.ids == ids


def test_search_documents():
    expected = [
        Document(page_content='Employees can work remotely.')
    ]
    vector_store = type(
        'FakeVectorStore',
        (),
        {
            'similarity_search': lambda self, **kwargs: (
                expected
                if kwargs == {
                    'query': 'remote work',
                    'k': 2,
                }
                else []
            )
        },
    )()

    results = search_documents(
        'remote work',
        vector_store,
        k=2,
    )

    assert results == expected
    