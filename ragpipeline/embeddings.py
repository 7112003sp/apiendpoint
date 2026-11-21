from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List,Any,Dict
from google import genai
from google.genai import types
import os
import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()
class Embedding():
    def __init__(self):
        self.chunk_size = 1000
        self.client = genai.Client(
                  api_key=os.getenv("GOOGLE_API_KEY")
               )
        self.model_name = "gemini-embedding-001"
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
        return self.embed_texts(texts)
    def embed_texts(self,texts):
        if not texts:
            raise ValueError("No text available for embeddings")
        response = self.client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
        output_dimensionality=768
    )
)
        embeddings = np.asarray(
            [item.values for item in response.embeddings],
        dtype=np.float32
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError(f"unexpected shape : {embeddings.shape}")
        return embeddings





        
    