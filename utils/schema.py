"""
Schema definitions for the FastAPI application.
"""
from pydantic import BaseModel


class ask_questionRequest(BaseModel):
    question: str


class ask_questionResponse(BaseModel):
    answer: str


class upload_pdfResponse(BaseModel):
    message: str


class delete_dbResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    error: str
