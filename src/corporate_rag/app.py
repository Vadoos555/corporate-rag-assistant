import logging
import streamlit as st

from corporate_rag.embeddings import create_embeddings
from corporate_rag.llm import create_chat_model
from corporate_rag.rag import answer_question, create_rag_chain
from corporate_rag.vector_store import create_vector_store
from corporate_rag.config import settings
from corporate_rag.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title='Corporate RAG Assistant',
    page_icon='📚',
    layout='centered'
)


@st.cache_resource
def load_rag_components():
    '''Create and cache the RAG components.'''
    embeddings = create_embeddings()
    vector_store = create_vector_store(
        embeddings=embeddings,
        persist_directory=settings.vector_store_dir
    )
    chat_model = create_chat_model()
    rag_chain = create_rag_chain(chat_model)
    
    return vector_store, rag_chain


def main() -> None:
    '''Render the corporate knowledge chatbot.'''
    st.title('📚 Corporate Knowledge Assistant')
    st.write(
        'Ask questions about company policies and information '
        'in the indexed PDF documents.'
    )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            if message.get('sources'):
                with st.expander('Sources'):
                    for source in message['sources']:
                        st.write(f'- {source}')
    
    question = st.chat_input('Ask a question about company documents...')
    
    if not question:
        return
    
    st.session_state.messages.append({
        'role': 'user',
        'content': question,
    })
    
    with st.chat_message('user'):
        st.markdown(question)
    
    with st.chat_message('assistant'):
        with st.spinner('Searching documents and preparing an answer...'):
            try:
                vector_store, rag_chain = load_rag_components()
                result = answer_question(question, vector_store, rag_chain)
                
                answer = result['answer']
                sources = result['sources']
                
                st.markdown(answer)
                
                if sources:
                    with st.expander('Sources'):
                        for source in sources:
                            st.write(f'- {source}')
                
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })
            except Exception as error:
                logger.exception('Failed to process the user question')
                st.error(
                    'The assistant could not process your question. '
                    'Check the API configuration and try again.'
                )


if __name__ == '__main__':
    main()
    