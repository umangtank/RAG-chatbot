import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


load_dotenv()

hf_token = os.getenv("HF_TOKEN")
ModelBaseDir = "efs_mount\model"
embedings = HuggingFaceBgeEmbeddings()
