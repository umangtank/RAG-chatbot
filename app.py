from langchain_community.utilities import SQLDatabase
from langchain_community.llms import Ollama
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

db = SQLDatabase.from_uri("sqlite:///mydatabase.db")
llm = Ollama(model = "llama2")


answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

print(chain.invoke({"question" : "how many unique products are there "}))