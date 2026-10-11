from langchain_openai import ChatOpenAI

from corporate_rag.config import settings


def create_chat_model() -> ChatOpenAI:
    '''Create the chat model used to generate answers.'''
    
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0
    )
    