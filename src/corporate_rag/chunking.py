
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    '''Create a text splitter with default chunk settings.'''
    
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    '''Split documents into smaller chunks.'''
    splitter = create_text_splitter()
    
    return splitter.split_documents(documents)
    