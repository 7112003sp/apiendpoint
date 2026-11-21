
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
    print("1. Request received", flush=True)
    pdf_bytes = await resume.read()
    print("2. PDF received", flush=True)

    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    print("3. PDF extracted", flush=True)

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
    print("5. Embeddings and index created", flush=True)

    search = RagSearch(faiss)

    result = search.summarise(
        jd,
        "Relevant technical skills, projects, work experience, education and qualifications"
    )
    print("7. AI analysis completed", flush=True)

    return {"message": result}
