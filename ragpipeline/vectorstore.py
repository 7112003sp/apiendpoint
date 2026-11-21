import faiss
import os
import numpy as np
from .embeddings import Embedding
import pickle
from typing_extensions import List,Any,Dict
from sentence_transformers import SentenceTransformer
class FaissStore:
    def __init__(self,model_name = "all-MiniLM-L6-v2"):
        self.client_dir = "./src/vectordb"
        os.makedirs(self.client_dir,exist_ok=True)
        self.index = None
        self.metadata = []
        self.model = SentenceTransformer(model_name)
        print(f"The Embedding Model [{self.model}] is loaded")
    def build_with_doc(self,docs:List[Any]):
        embedded_pipeline = Embedding()
        chunk_with_pdf = embedded_pipeline.chunk_pdf(docs)
        metadatas = [{"Test":chunk.page_content} for chunk in chunk_with_pdf]
        embeddings = embedded_pipeline.embedding_genrator(chunk_with_pdf)
        self.add_embeddings(np.array(embeddings).astype(np.float32),metadatas)
        self.save()
    def add_embeddings(self,embeddings:np.ndarray,metadata:List[Any]):
        dimensions = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimensions)
        self.index.add(
            embeddings
        )
        if metadata:
            self.metadata.extend(metadata)
    def save(self):
        faiss_path = os.path.join(self.client_dir,"faiss.index")
        meta_path = os.path.join(self.client_dir,"metadata.pkl")
        faiss.write_index(self.index,faiss_path)
        with open(meta_path,"wb") as f:
            pickle.dump(self.metadata,f)
        print("Sucessefully Saved")
    def load(self):
        faiss_path = os.path.join(self.client_dir,"faiss.index")
        meta_path = os.path.join(self.client_dir,"metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path,"rb") as f:
            self.metadata = pickle.load(f)
        print("Data is loaded Successfully")
    def search(self,query_embdding,top_k=3):
        D,I = self.index.search(query_embdding,top_k)
        res = []
        for idx,dist in zip(I[0],D[0]):
            meta = self.metadata[idx] if  0 <= idx < len(self.metadata) else None
            res.append({"index":idx,"distance":dist,"meta":meta})
        return res
    def query(self,query):
        query_em = self.model.encode([query]).astype(np.float32)
        res = self.search(query_em)
        return res



            

                 
        

