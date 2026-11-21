from fastapi import FastAPI
from langchain_groq import ChatGroq
from pydantic import BaseModel
import os
import dotenv
from dotenv import load_dotenv
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b"
)
class Example(BaseModel):
    prompt:str
app = FastAPI()
@app.get('/')
def geting():
    return {"name":"hi"}
@app.post("/post")
def posting(name:Example):
    resp = llm.invoke(name.prompt+"give response in two lines")
    return {
        "message":"user created",
        "query":name.prompt,
        "response":resp
    }


