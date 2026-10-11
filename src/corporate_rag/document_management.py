from langchain_chroma import Chroma


def delete_document(source: str, vector_store: Chroma) -> int:
    '''Delete all indexed chunks belonging to one source file.'''
    if not source.strip():
        raise ValueError('Document source cannot be empty.')
    
    result = vector_store.get(where={'source': source})
    ids = result.get('ids', [])
    
    if ids:
        vector_store.delete(ids=ids)
    
    return len(ids)


def clear_vector_store(vector_store: Chroma) -> int:
    '''Delete all indexed chunks from the vector store.'''
    result = vector_store.get()
    ids = result.get('ids', [])
    
    if ids:
        vector_store.delete(ids=ids)
    
    return len(ids)
