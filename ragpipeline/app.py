from dataloader import loadPdf
from vectorstore import FaissStore
from search import RagSearch
if __name__ == "__main__":
    docs   = loadPdf("./pdfs")
    faiss = FaissStore()
    faiss.build_with_doc(docs=docs)
    faiss.load()
    print(faiss.query("who is surya"))
    search = RagSearch()
    res = search.summarise("who is suryaprakash")
    print(res)
    