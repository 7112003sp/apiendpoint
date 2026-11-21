from langchain_groq import ChatGroq
from .vectorstore import FaissStore
import os
from .dataloader import loadPdf
from dotenv import load_dotenv
from .structuredoutput import ResponseModel

load_dotenv()

class RagSearch:
    def __init__(self, vectorStore):
        self.vectorStore = vectorStore
        self.llm = ChatGroq(model="openai/gpt-oss-120b")
    def summarise(self,jd,query):
        res = self.vectorStore.query(query)
        texts = [r["meta"].get("Test","") for r in res if r["meta"]]
        context = "\n\n\n".join(texts)
        if not context:
            return "There is no context"
        prompt = f"""
You are an expert resume analyzer.

Resume Context:
{context}

Job Description:
{jd}

Tasks:
1. Compare the resume with the job description.
2. Identify matching skills.
3. Identify missing skills.
4. Give a match score between 0 and 100.
5. Suggest improvements.

Use only information supported by the resume.
Do not invent experience or skills.
"""
        structured = self.llm.with_structured_output(ResponseModel)
        response = structured.invoke(prompt)
        return response.model_dump()
       


        
