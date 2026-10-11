
from corporate_rag.chunking import split_documents
from corporate_rag.config import settings
from corporate_rag.embeddings import create_embeddings
from corporate_rag.indexing import index_documents
from corporate_rag.ingestion import load_pdfs, validate_documents
from corporate_rag.retrieval import search_documents
from corporate_rag.vector_store import create_vector_store


def main() -> None:
    '''Load PDFs, index their chunks, and run a sample search.'''
    documents = validate_documents(load_pdfs(settings.pdf_dir))
    if not documents:
        raise ValueError(f'No readable PDF pages found in {settings.pdf_dir}')

    chunks = split_documents(documents)
    print(f'Loaded pages: {len(documents)}')
    print(f'Created chunks: {len(chunks)}')

    embeddings = create_embeddings()
    vector_store = create_vector_store(
        embeddings=embeddings,
        persist_directory=settings.vector_store_dir,
    )

    ids = index_documents(chunks, vector_store)
    print(f'Indexed chunks: {len(ids)}')

    results = search_documents(
        query='What is the remote work policy?',
        vector_store=vector_store,
        k=settings.retrieval_k,
    )

    print('\nSample search results:')
    for document in results:
        print(f"\nSource: {document.metadata.get('source', 'Unknown')}")
        print(document.page_content)


if __name__ == '__main__':
    main()
    