from contextlib import contextmanager
import sys


@contextmanager
def nullify_output(suppress_stdout=True, suppress_stderr=True):
    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr

from pydantic import BaseModel
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough
)
from langchain_core.documents import Document

import uuid
import pickle
from itertools import chain

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pdf_paths = ["./AMD.10K.2023.pdf", "./IBM.10K.2023.pdf", "./AAPL.10K.2023.pdf"]
pdfs = ["AMD.10K.2023.pdf", "IBM.10K.2023.pdf", "AAPL.10K.2023.pdf"]

raw_pdf_elements = []
pickle_paths = ["./assets/AMD.10K.2023.pdf-0.pkl", "./assets/IBM.10K.2023.pdf-2.pkl", "./assets/AAPL.10K.2023.pdf-4.pkl"]
for pdf in pickle_paths:
    with open(f"{pdf}", 'rb') as f:
        raw_pdf_elements.append(pickle.load(f))


class Element(BaseModel):
    type: str
    text: Any


# Categorize by type
categorized_elements = [
    [
        Element(type="table", text=str(element.metadata.text_as_html))
        if "unstructured.documents.elements.Table" in str(type(element))
        else Element(type="text", text=str(element))
        for element in raw_pdf_element
    ]
    for raw_pdf_element in raw_pdf_elements
]

table_elements = [[e for e in categorized_element if e.type == "table"] for categorized_element in categorized_elements]
text_elements = [[e for e in categorized_element if e.type == "text"] for categorized_element in categorized_elements]


def get_docs(text_ele):
    pdf_docs = []
    pdf_docs.extend(
        [Document(page_content=ele.text, metadata={"pdf_title": t[1]}) for ele in t[0]] for i, t in
        enumerate(zip(text_ele, pdfs))
    )
    pdf_docs = list(chain(*pdf_docs))
    return pdf_docs


table_docs = get_docs(table_elements)
text_docs = get_docs(text_elements)

with open("./assets/table_summaries-3.pkl", 'rb') as f:
    table_summaries = pickle.load(f)

with open("./assets/text_summaries.pkl", 'rb') as f:
    text_summaries = pickle.load(f)

text_ids = [str(uuid.uuid4()) for _ in text_docs]
table_ids = [str(uuid.uuid4()) for _ in table_docs]

id_key = "doc_id"

text_summaries_docs = [
    Document(page_content=text_summaries[i],
             metadata={id_key: text_ids[i], "pdf_title": text_doc.metadata['pdf_title']})
    for i, text_doc in enumerate(text_docs)
]
table_summaries_docs = [
    Document(page_content=table_summaries[i],
             metadata={id_key: table_ids[i], "pdf_title": table_doc.metadata['pdf_title']})
    for i, table_doc in enumerate(table_docs)
]

vectorstore = ElasticsearchStore(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    es_url="http://localhost:9200",
    index_name="summaries_index",
    strategy=ElasticsearchStore.ApproxRetrievalStrategy()
)

with nullify_output(suppress_stdout=True, suppress_stderr=True):
    vectorstore.add_documents(text_summaries_docs)
    vectorstore.add_documents(table_summaries_docs)

docs_w_ids = list(zip(text_ids + table_ids, text_docs + table_docs))


def get_orig(summary_docs):
    out_docs = [docs[1] for summary_doc in summary_docs for docs in docs_w_ids if
                docs[0] == summary_doc.metadata[id_key]]
    return out_docs


model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106")

get_pdf_query = """You are an assistant tasked with generating additional questions from the given query. \
Given a set of questions, give the relevant questions (in the format as shown) pertaining to each individual company \
in the query IF there are more than one. Also give the report name it corresponds to.
Report names:
AMD.10K.2023.pdf
AAPL.10K.2023.pdf
IBM.10K.2023.pdf
CSCO.10K.2023.pdf
UBER.10K.2023.pdf

<--example start-->
Query: What are the equity compensation plans of AMD and Cisco?
Answer:
What are the equity compensation plans of AMD?, AMD.10K.2023.pdf
What are the equity compensation plans of Cisco?, CSCO.10K.2023.pdf
<--example end-->

<--example start-->
Are there any ongoing legal disputes with Uber?
Answer:
Are there any ongoing legal disputes with Uber?, UBER.10K.2023.pdf
<--example end-->

Query: {user_query}
Answer:
"""
get_pdf_query_prompt = ChatPromptTemplate.from_template(get_pdf_query)
get_pdf_query_chain = {"user_query": RunnablePassthrough()} | get_pdf_query_prompt | model | StrOutputParser()


def get_context(pdf_response):
    context_out = []
    for resp in pdf_response.split('\n'):
        context_out.append(
            get_orig(
                vectorstore.similarity_search(resp.split(',')[0], k=3, filter=[
                    {"term": {"metadata.pdf_title.keyword": resp.split(',')[1].strip()}}])
            )
        )

    return context_out


def parse_context(contexts):
    contexts = list(chain(*contexts))
    str_out = ""
    for context in contexts:
        str_out += "CONTEXT FROM " + context.metadata['pdf_title'] + "\n"
        str_out += context.page_content + "\n\n"

    return str_out


context_chain = get_pdf_query_chain | get_context | parse_context

rag_prompt_text = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question \
in as many words as required.
Feel free to go into the details of what's presented in the context down below.
If you don't know the answer, just say "I don't know."
Question: {question}
Context: {context}
Answer: 
"""

rag_prompt = ChatPromptTemplate.from_template(rag_prompt_text)

rag_chain = (
        {"question": RunnablePassthrough(), "context": context_chain}
        | rag_prompt
        | model
        | StrOutputParser()
)


def main():
    while True:
        user_input = input("Enter input (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        with nullify_output(suppress_stdout=True, suppress_stderr=True):
            result = rag_chain.invoke(user_input)
        print("MODEL OUTPUT: ", result)


if __name__ == "__main__":
    main()
