from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List,Any,Dict
class Embedding():
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"
        self.model = SentenceTransformer(self.model_name)
        self.chunk_size = 1000
        self.overlap = 200
    def chunk_pdf(self,docs:List[Any]):
        chunked = RecursiveCharacterTextSplitter(
            chunk_size = self.chunk_size,
            chunk_overlap = self.overlap,
            length_function = len,
            separators=["\n\n\n","\n"," ",""]
        )
        chunked_pdf = chunked.split_documents(docs)
        return chunked_pdf
    def embedding_genrator(self,chunked:List[Any]):
        texts = [chunks.page_content for chunks in chunked]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )
        return embeddings




        
    