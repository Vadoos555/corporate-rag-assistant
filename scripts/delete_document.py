import argparse

from corporate_rag.config import settings
from corporate_rag.document_management import delete_document
from corporate_rag.embeddings import create_embeddings
from corporate_rag.vector_store import create_vector_store


def main() -> None:
    '''Delete one PDF source from the vector index.'''
    parser = argparse.ArgumentParser(
        description='Delete a document from the Chroma index.'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Exact source path stored in document metadata.',
    )
    args = parser.parse_args()

    vector_store = create_vector_store(
        embeddings=create_embeddings(),
        persist_directory=settings.vector_store_dir,
    )

    deleted_count = delete_document(args.source, vector_store)
    print(f'Deleted chunks: {deleted_count}')


if __name__ == '__main__':
    main()
    