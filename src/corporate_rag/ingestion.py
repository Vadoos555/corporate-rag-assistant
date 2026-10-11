from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from corporate_rag.config import settings


def load_pdf(file_path: Path) -> list[Document]:
    """Load text and metadata from a validate PDF file"""
    if not file_path.exists():
        raise FileNotFoundError(f'PDF file not found: {file_path}')
    
    if not file_path.is_file():
        raise ValueError(f'Path is not file: {file_path}')
    
    if file_path.suffix.lower() != '.pdf':
        raise ValueError(f'Expected a PDF file: {file_path}')
    
    max_size = settings.max_pdf_size_mb * 1024 * 1024
    if file_path.stat().st_size > max_size:
        raise ValueError(
            f'PDF exceeds the maximum size of '
            f'{settings.max_pdf_size_mb} MB.'
        )
    
    with file_path.open('rb') as pdf_file:
        signature = pdf_file.read(5)

    if signature != b'%PDF-':
        raise ValueError(f'File does not have a valid PDF signature: {file_path}')

    return PyPDFLoader(str(file_path)).load()


def load_pdfs(pdf_dir: Path) -> list[Document]:
    '''Load all PDF files from a directory.'''
    if not pdf_dir.exists():
        raise FileNotFoundError(f'PDF directory is not found: {pdf_dir}')
    
    if not pdf_dir.is_dir():
        raise ValueError(f'Path is not a directory: {pdf_dir}')
    
    pdf_files = sorted(pdf_dir.glob('*.pdf'))
    
    documents = []
    for pdf_file in pdf_files:
        documents.extend(load_pdf(pdf_file))
    
    return documents


def validate_documents(documents: list[Document]) -> list[Document]:
    '''Remove documents that contain no readable text.'''
    valid_documents = []
    
    for document in documents:
        if document.page_content.strip():
            valid_documents.append(document)
    
    return valid_documents
