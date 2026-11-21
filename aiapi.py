from fastapi import FastAPI,UploadFile,File,Form
import fitz
from langchain_core.documents import Document
app = FastAPI()

@app.post("/analyse")
async def analyse(
    resume:UploadFile = File(...),
    jd:str = Form(...)
):
    from ragpipeline.vectorstore import FaissStore
    from ragpipeline.search import RagSearch
    print(resume.filename)
    print(jd)
    pdf_bytes = await resume.read()
    pdf = fitz.open(stream=pdf_bytes,filetype="pdf")
    docs = []
    for page_number,page in enumerate(pdf):
        text = page.get_text()
        docs.append(
           Document(
                page_content = text,
                metadata = {"page":page_number + 1}
           )
        )
    pdf.close()
    faiss = FaissStore()
    faiss.build_with_doc(docs=docs)
    print(faiss.query("who is surya"))
    search = RagSearch(faiss)
    res = search.summarise(jd,"who is suryaprakash")
    print(res)

    return {"message":res}
