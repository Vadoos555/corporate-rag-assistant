
from unittest.mock import patch

from corporate_rag.embeddings import create_embeddings
from corporate_rag.llm import create_chat_model


def test_create_embeddings_uses_settings():
    with (
        patch('corporate_rag.embeddings.settings') as mock_settings,
        patch('corporate_rag.embeddings.OpenAIEmbeddings') as mock_model,
    ):
        mock_settings.embedding_model = 'test-embedding-model'
        mock_settings.openai_api_key = 'test-key'

        create_embeddings()

    mock_model.assert_called_once_with(
        model='test-embedding-model',
        api_key='test-key',
    )


def test_create_chat_model_uses_settings():
    with (
        patch('corporate_rag.llm.settings') as mock_settings,
        patch('corporate_rag.llm.ChatOpenAI') as mock_model,
    ):
        mock_settings.chat_model = 'test-chat-model'
        mock_settings.openai_api_key = 'test-key'

        create_chat_model()

    mock_model.assert_called_once_with(
        model='test-chat-model',
        api_key='test-key',
        temperature=0,
    )