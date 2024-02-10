from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

pdf_names = {"AMD.10K.2023.pdf": "10K report of AMD",
             "BABA.10K.2023.pdf": "10K report of Alibaba",
             "CSCO.10K.2023.pdf": "10K report of Cisco",
             "IBM.10K.2023.pdf": "10K report of IBM",
             "UBER.10K.2023.pdf": "10K report of Uber"}

get_split_ques_text = """You are an assistant tasked with generating additional questions according to the given instructions. \
Given a query, generate question(s) so as to query from the individual companies, but do NOT change the \
meaning and keywords of the original query. Separate each split question by a newline. In the question, include the \
company name only and NOT '10K report by xxx'.
<--example start-->
Query: What are the equity compensation plans of AMD and Cisco?
Answer:
What is the equity compensation plans of AMD?
What is the equity compensation plans of Cisco?
<--example end-->

Query: {user_query}
Answer:
"""
get_split_ques_prompt = ChatPromptTemplate.from_template(get_split_ques_text)

def get_pdfs_from_chain(resp):
  pdf_list = ""
  for pdf in resp.split(','):
    pdf_list += f"""{pdf.strip()}: {pdf_names[f'{pdf.strip()}']}\n"""
  return pdf_list

get_ques_chain = get_split_ques_prompt | model | StrOutputParser()

get_pdfs_to_query_text = """You are an assistant tasked with extracting the PDFs to query from the given query. \
Given a set of questions, give the relevant PDFs (separated by commas) from the list that can be used to answer the questions.
<--example start-->
Query:
What is the revenue of IBM for the year 2022?
What is the revenue of Uber for the year 2022?
PDFs:
AMD.10K.2023.pdf - 10K report of AMD
IBM.10K.2023.pdf - 10K report of IBM
BABA.10K.2023.pdf - 10K report of AliBaba
CSCO.10K.2023.pdf - 10K report of Cisco
UBER.10K.2023.pdf - 10K report of Uber
Answer: IBM.10K.2023.pdf, UBER.10K.2023.pdf
<--example end-->

Query: {user_query}
PDFs:
{pdf_names}
Answer:\n"""
get_pdfs_to_query_prompt = ChatPromptTemplate.from_template(get_pdfs_to_query_text)

get_pdfs_chain = get_pdfs_to_query_prompt | model | StrOutputParser()
