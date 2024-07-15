import os

from dotenv import load_dotenv


load_dotenv()

hf_token = os.getenv("HF_TOKEN")
ModelBaseDir = "efs_mount\model"
