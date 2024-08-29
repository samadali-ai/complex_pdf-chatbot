import os
from dotenv import load_dotenv
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from parser import load_or_parse_data
##### LLAMAPARSE #####
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#
from groq import Groq
from langchain_groq import ChatGroq
from parser import load_or_parse_data
from vector_database import create_vector_database
import joblib
import nest_asyncio  # noqa: E402
nest_asyncio.apply()


chat_model = ChatGroq(temperature=0,
                      model_name="mixtral-8x7b-32768",
                      api_key=os.environ.get("GROQ_API_KEY"),)


embed_model = HuggingFaceEmbeddings(model_name="bert-base-uncased")  # Change model_name if needed

vectorstore = Chroma(embedding_function=embed_model,
                      persist_directory="chroma_db_llamaparse1",
                      collection_name="rag")
 #
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#
prompt = set_custom_prompt()
prompt


qa = RetrievalQA.from_chain_type(llm=chat_model,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})
response = qa.invoke({"query": "what is the Balance of UBER TECHNOLOGIES, INC.as of December 31, 2021?"})
response['result']
