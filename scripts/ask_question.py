from pathlib import Path

from corporate_rag.embeddings import create_embeddings
from corporate_rag.llm import create_chat_model
from corporate_rag.rag import answer_question, create_rag_chain
from corporate_rag.vector_store import create_vector_store


def main() -> None:
    '''Run the RAG question-answering workflow.'''
    embeddings = create_embeddings()

    vector_store = create_vector_store(
        embeddings=embeddings,
        persist_directory=Path('data/vector_store'),
    )

    chat_model = create_chat_model()
    rag_chain = create_rag_chain(chat_model)

    question = input('Ask a question about company documents: ').strip()

    if not question:
        print('Please enter a question.')
        return

    result = answer_question(
        question=question,
        vector_store=vector_store,
        rag_chain=rag_chain,
    )

    print('\nAnswer:')
    print(result['answer'])

    print('\nSources:')
    for source in result['sources']:
        print(f'- {source}')


if __name__ == '__main__':
    main()
    