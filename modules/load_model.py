from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

from utils import constants


class LoadMOdel:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                constants.ModelBaseDir, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                constants.ModelBaseDir, trust_remote_code=True
            )
        except Exception as exc:
            return str(exc)

    def load_llm(self):
        try:
            generator = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=True,
                temperature=0.7,
                max_new_tokens=512,
                repetition_penalty=1.1,
                do_sample=True,
            )
            self.llm = HuggingFacePipeline(pipeline=generator)
            return self.llm
        except Exception as exc:
            raise exc
