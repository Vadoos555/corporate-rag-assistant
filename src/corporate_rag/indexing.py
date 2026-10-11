import hashlib

from langchain_core.documents import Document
from langchain_chroma import Chroma


def index_documents(documents: list[Document], vector_store: Chroma) -> list[str]:
    '''Add document chunk to the vector store.'''
    if not documents:
        raise ValueError('No documents to index.')
    
    ids = []
    
    for document in documents:
        source = str(document.metadata.get('source', 'unknown'))
        page = document.metadata.get('page', 'unknown')
        start_index = document.metadata.get('start_index', 'unknown')
        
        unigue_text = (f'{source}|{page}|{start_index}|{document.page_content}')
        document_id = hashlib.sha256(unigue_text.encode('utf-8')).hexdigest()
        ids.append(document_id)
    
    vector_store.add_documents(documents=documents, ids=ids)
    
    return ids
