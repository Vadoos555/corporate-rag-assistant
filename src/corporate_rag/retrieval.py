from langchain_core.documents import Document
from langchain_chroma import Chroma


def search_documents(query: str, vector_store: Chroma, k: int = 4) -> list[Document]:
    '''Find the most relevant document chunks for a query.'''
    if not query.strip():
        raise ValueError('Query cannot be empty.')
    
    if k < 1:
        raise ValueError('k must be at least 1.')
    
    return vector_store.similarity_search(query=query, k=k)
