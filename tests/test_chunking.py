
from langchain_core.documents import Document

from corporate_rag.chunking import create_text_splitter
from corporate_rag.chunking import split_documents


def test_create_text_splitter():
    splitter = create_text_splitter()
    text = 'This is a test sentence. ' * 100

    chunks = splitter.split_text(text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_split_documents():
    text = 'Employee handbook. ' * 200
    documents = [
        Document(
            page_content=text,
            metadata={'source': 'handbook.pdf', 'page': 0},
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata['source'] == 'handbook.pdf' for chunk in chunks)
    assert all('start_index' in chunk.metadata for chunk in chunks)


def test_chunks_preserve_page_metadata():
    documents = [
        Document(
            page_content='Remote work policy. ' * 100,
            metadata={'source': 'handbook.pdf', 'page': 2},
        )
    ]

    chunks = split_documents(documents)

    assert all(chunk.metadata['source'] == 'handbook.pdf' for chunk in chunks)
    assert all(chunk.metadata['page'] == 2 for chunk in chunks)


def test_short_document_creates_one_chunk():
    documents = [
        Document(
            page_content='Short policy text.',
            metadata={'source': 'handbook.pdf', 'page': 0},
        )
    ]

    chunks = split_documents(documents)

    assert len(chunks) == 1
    assert chunks[0].page_content == 'Short policy text.'