"""
Simple FastAPI Application for Question-Answering System with PDF Upload
Author: Umang Tank
"""
import os

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from modules import load_model  # Assuming correct import path
from utils import common
from utils import schema

app = FastAPI()


@app.get("/")
def read_root():
    """
    Root endpoint to verify if all models are initialized.
    """
    return "All models initialized"


@app.post("/AskQA")
def ask_question(request: schema.QuestionRequest):
    """
    Endpoint to ask a question and get an answer from the loaded model.

    Args:
        request (schema.QuestionRequest): Request body containing the question.

    Returns:
        dict: Response containing the answer or error message.
    """
    try:
        model = load_model.LoadMOdel()
        question = request.question
        answer = model.load_llm().invoke(question)
        return {"Answer": answer}
    except Exception as error:
        return {"Error": str(error)}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file.

    Args:
        file (UploadFile): PDF file uploaded by the user.

    Returns:
        JSONResponse: JSON response indicating success or failure.
    """
    try:
        print(file.filename)
        contents = await file.read()
        print(contents)
        common.CreateDB(contents, file)

        return JSONResponse(
            content={"message": "File uploaded successfully and chunks created"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
