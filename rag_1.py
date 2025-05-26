from typing import Literal
import os

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "LangChain/LocalRAG-App"

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import PromptTemplate
from typing_extensions import Annotated, List, TypedDict
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore

llm = ChatOllama(
    model="gemma3:4b",
    base_url="http://localhost:11434",
)

embeddings = OllamaEmbeddings(
    model="llama3",
    base_url="http://localhost:11434"
)


vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(all_splits)


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

question = "When was Nike incorporated?"

retrieved_docs = vector_store.similarity_search(question)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
# docs_content = """
# Table of Contents
# INTERNATIONAL MARKETS
# Nike was incorporated on January 25, 1964, as Blue Ribbon Sports, Inc. and became Nike, Inc. on May 30, 1971.
# For fiscal 2023, non-U.S. NIKE Brand and Converse sales accounted for approximately 57% of total revenues, compared to 60% and 61% for fiscal 2022 and fiscal 2021,
# respectively. We sell our products to retail accounts through our own NIKE Direct operations and through a mix of independent distributors, licensees and sales
# representatives around the world. We sell to thousands of retail accounts and ship products from 67 distribution centers outside of the United States. Refer to Item 2.
# Properties for further information on distribution facilities outside of the United States. During fiscal 2023, NIKE's three largest customers outside of the United States
# accounted for approximately 14% of total non-U.S. sales.
# In addition to NIKE-owned and Converse-owned digital commerce platforms in over 40 countries, our NIKE Direct and Converse direct to consumer businesses operate
# the following number of retail stores outside the United States:

# leisure footwear companies, athletic and leisure apparel companies, sports equipment companies and large companies having diversified lines of athletic and leisure
# footwear, apparel and equipment, including adidas, Anta, ASICS, Li Ning, lululemon athletica, New Balance, Puma, Under Armour and V.F. Corporation, among others.
# The intense competition and the rapid changes in technology and consumer preferences in the markets for athletic and leisure footwear and apparel and athletic
# equipment constitute significant risk factors in our operations. Refer to Item 1A. Risk Factors for additional information.
# NIKE is the largest seller of athletic footwear and apparel in the world. Important aspects of competition in this industry are:
# • Product attributes such as quality; performance and reliability; new product style, design, innovation and development; as well as consumer price/value.

# income (loss), Long-term debt or Net income depending on the nature of the underlying exposure, whether the derivative is formally designated as a hedge and, if
# designated, the extent to which the hedge is effective. The Company classifies the cash flows at settlement from derivatives in the same category as the cash flows from
# the related hedged items. For undesignated hedges and designated cash flow hedges, this is primarily within the Cash provided by operations component of the
# Consolidated Statements of Cash Flows. For designated net investment hedges, this is within the Cash provided by investing activities component of the Consolidated
# Statements of Cash Flows. For the Company's fair value hedges, which are interest rate swaps used to mitigate the change in fair value of its fixed-rate debt attributable
# to changes in interest rates, the related cash flows from periodic interest payments are reflected within the Cash provided by operations component of the Consolidated

# Heidi O'Neill, President, Consumer, Brand & Product — Ms. O'Neill, 58, joined NIKE in 1998 and leads the integration of global Men's,Women's & Kids' consumer teams, the entire global product engine and global brand marketing and sports marketing to build deepstorytelling, relationships and engagement with the brand. Since joining NIKE, she has held a variety of key roles, including leadingNIKE's marketplace and four geographic operating regions, leading NIKE Direct and accelerating NIKE's retail and digital-commercebusiness and creating and leading NIKE's Women’s business. Prior to NIKE, Ms. O'Neill held roles at Levi Strauss & Company and Foote,Cone & Belding.
# """
# print(docs_content)
# docs_content = "Nike was incorporated on January 25, 1964, as Blue Ribbon Sports, Inc. and became Nike, Inc. on May 30, 1971."
print(docs_content)
# prompt = prompt.invoke({"question": question, "context": docs_content})
# print(prompt)
# print(prompt)
# print(len(docs_content))
# answer = llm.invoke(prompt)
# print(answer)

# answer = llm.invoke("Hey there!")
# print(answer)