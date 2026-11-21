
from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI()


@app.get("/")
def health():
    return {"status": "FastAPI running"}


@app.post("/analyse")
async def analyse(
    resume: UploadFile = File(...),
    jd: str = Form(...)
):
    import fitz
    from langchain_core.documents import Document
    from ragpipeline.vectorstore import FaissStore
    from ragpipeline.search import RagSearch

    pdf_bytes = await resume.read()

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    docs = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        docs.append(
            Document(
                page_content=text,
                metadata={"page": page_number + 1}
            )
        )

    pdf.close()

    faiss = FaissStore()
    faiss.build_with_doc(docs=docs)

    search = RagSearch(faiss)

    result = search.summarise(
        jd,
        "Relevant technical skills, projects, work experience, education and qualifications"
    )

    return {"message": result}
