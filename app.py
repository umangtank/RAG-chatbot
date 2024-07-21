"""
Simple FastAPI Application for Question-Answering System with PDF Upload
Author: Umang Tank
"""
import os

import fastapi

from modules import load_model  # Assuming correct import path
from utils import common
from utils import schema

app = fastapi.FastAPI()


@app.get("/")
def read_root() -> str:
    """
    Root endpoint to verify if all models are initialized.

    Returns:
        str: Confirmation message indicating models are initialized.
    """
    return "All models initialized"


@app.post("/AskQA")
def ask_question(request: schema.ask_questionRequest) -> schema.ask_questionResponse:
    """
    Endpoint to ask a question and get an answer from the loaded model.

    Args:
        request (schema.ask_questionRequest): Request body containing the question.

    Returns:
        schema.ask_questionResponse: Response containing the answer or error message.
    """
    try:
        model = load_model.LoadMOdel()
        question = request.question
        llm = model.load_llm()
        answer = common.ChainCreation(llm)({"query": question})
        return schema.ask_questionResponse(answer=answer)
    except Exception as exc:
        return fastapi.JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/upload-pdf/")
async def upload_pdf(
    file: fastapi.UploadFile = fastapi.File(...),
) -> schema.upload_pdfResponse:
    """
    Endpoint to upload a PDF file.

    Args:
        file (UploadFile): PDF file uploaded by the user.

    Returns:
        schema.upload_pdfResponse: JSON response indicating success or failure.
    """
    try:
        print(file.filename)
        contents = await file.read()
        print(contents)
        common.CreateDB(contents, file)

        return schema.upload_pdfResponse(
            message="File uploaded successfully and chunks created"
        )
    except Exception as exc:
        return fastapi.JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/DeleteDB")
def delete_db() -> schema.delete_dbResponse:
    """
    Endpoint to delete the database files.

    Returns:
        schema.delete_dbResponse: Response indicating the database was deleted successfully.
    """
    try:
        os.remove("faiss_index")
        os.remove("documents")
        return schema.delete_dbResponse(message="Database deleted successfully")
    except Exception as exc:
        return fastapi.JSONResponse(status_code=500, content={"error": str(exc)})
