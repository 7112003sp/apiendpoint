from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from .vectorstore import FaissStore
from langchain_core.documents import Document
def loadPdf(pdf_directory):
    try:
        all_docs_data = []
        pdf_directory_path = Path(pdf_directory)
        pdf_files = pdf_directory_path.glob("**/*.pdf")
        for pdf_file in pdf_files:
            print(f"[{pdf_file}] loading pdf file...")
            file_loader = PyMuPDFLoader(pdf_file)
            loaded_document = file_loader.load()
            all_docs_data.extend(loaded_document)
        print(f"Successfully loaded pdf files from {pdf_directory_path}")
        return all_docs_data
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"There is an error {e}")



