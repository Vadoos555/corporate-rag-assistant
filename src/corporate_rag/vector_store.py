from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


def create_vector_store(embeddings: OpenAIEmbeddings, persist_directory: Path) -> Chroma:
    '''Create or open a persistent Chroma vector store.'''
    persist_directory.mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        collection_name='corporate_documents',
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    