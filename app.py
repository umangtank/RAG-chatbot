from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama


db = SQLDatabase.from_uri("sqlite:///mydatabase.db")
llm = Ollama(model = "llama2")

print(llm.invoke("hello, who are you ?"))
