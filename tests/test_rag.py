
import pytest

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from corporate_rag.rag import (
    answer_question,
    create_rag_chain,
    format_context,
)


def test_format_context():
    documents = [
        Document(
            page_content='Remote work is allowed.',
            metadata={'source': 'handbook.pdf', 'page': 0},
        ),
    ]

    context = format_context(documents)

    assert 'Remote work is allowed.' in context
    assert 'handbook.pdf, page 1' in context


def test_answer_question(monkeypatch):
    documents = [
        Document(
            page_content='Remote work is allowed.',
            metadata={'source': 'handbook.pdf', 'page': 0},
        ),
    ]

    def fake_search_documents(query, vector_store, k):
        assert query == 'Can employees work remotely?'
        assert k == 4
        return documents

    monkeypatch.setattr(
        'corporate_rag.rag.search_documents',
        fake_search_documents,
    )

    chat_model = RunnableLambda(
        lambda prompt: 'Yes, remote work is allowed.'
    )
    rag_chain = create_rag_chain(chat_model)

    result = answer_question(
        question='Can employees work remotely?',
        vector_store=object(),
        rag_chain=rag_chain,
    )

    assert result['answer'] == 'Yes, remote work is allowed.'
    assert result['sources'] == ['handbook.pdf']
    assert result['documents'] == documents


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match='Question cannot be empty'):
        answer_question(
            question='   ',
            vector_store=object(),
            rag_chain=object(),
        )



def test_no_documents_returns_fallback(monkeypatch):
    def fake_search_documents(query, vector_store, k):
        return []

    def fail_if_called(_):
        raise AssertionError('LLM must not be called without documents.')

    monkeypatch.setattr(
        'corporate_rag.rag.search_documents',
        fake_search_documents,
    )

    result = answer_question(
        question='What is the vacation policy?',
        vector_store=object(),
        rag_chain=fail_if_called,
    )

    assert result['answer'] == (
        'I could not find relevant information in the documents.'
    )
    assert result['sources'] == []
    assert result['documents'] == []
    