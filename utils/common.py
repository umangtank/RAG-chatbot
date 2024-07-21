import os

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from utils import constants


def CreateDB(contents, file):
    UPLOAD_DIR = "./documents"
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    db = FAISS.from_documents(documents, constants.embedings)
    db.save_local("faiss_index")


def ChainCreation(llm):
    db = FAISS.load_local("faiss_index", constants.embedings)
    Retriever = db.as_retriever(search_type="similarity")
    chain = RetrievalQA.from_chain_type(llm, retriever=Retriever)
