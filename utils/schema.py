from pydantic import BaseModel
from pydantic import ValidationError


class QuestionRequest(BaseModel):
    question: "str"
