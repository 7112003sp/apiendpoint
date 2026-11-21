from pydantic import BaseModel,Field
from typing_extensions import List

class ResponseModel(BaseModel):
    score:int = Field(
        ge=0,
        le = 100,
        description="Resume match score between 0 - 100"
    )
    matching_skills:List[str] = Field(
        description="Skilss present in the both resume and jd"
    )
    missing_skills:List[str] = Field(
        description="Skills missing from resume which are in jd"
    )
    suggestions:str = Field(
        description="Suggestions to improve resume"
    )
    summary:str = Field(
        description="Overall Resume analysis"
    )
