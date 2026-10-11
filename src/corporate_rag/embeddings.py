from langchain_openai import OpenAIEmbeddings

from corporate_rag.config import settings


def create_embeddings() -> OpenAIEmbeddings:
    '''Create the embedding model for document indexing and search.'''
    
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        )
