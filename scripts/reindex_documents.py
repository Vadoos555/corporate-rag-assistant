
from corporate_rag.chunking import split_documents
from corporate_rag.config import settings
from corporate_rag.document_management import clear_vector_store
from corporate_rag.embeddings import create_embeddings
from corporate_rag.indexing import index_documents
from corporate_rag.ingestion import load_pdfs, validate_documents
from corporate_rag.vector_store import create_vector_store


def main() -> None:
    '''Rebuild the vector index from all PDF files.'''
    documents = validate_documents(load_pdfs(settings.pdf_dir))

    if not documents:
        raise ValueError(
            f'No readable PDF documents found in {settings.pdf_dir}'
        )

    chunks = split_documents(documents)
    embeddings = create_embeddings()

    vector_store = create_vector_store(
        embeddings=embeddings,
        persist_directory=settings.vector_store_dir,
    )

    removed_count = clear_vector_store(vector_store)
    ids = index_documents(chunks, vector_store)

    print(f'Removed old chunks: {removed_count}')
    print(f'Loaded pages: {len(documents)}')
    print(f'Created chunks: {len(chunks)}')
    print(f'Indexed chunks: {len(ids)}')


if __name__ == '__main__':
    main()
    