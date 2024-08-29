import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from parser import load_or_parse_data
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Load environment variables from .env file
load_dotenv()

def create_vector_database():
    try:
        llama_parse_documents = load_or_parse_data()
        output_dir = 'data'
        output_file = os.path.join(output_dir, 'output.md')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'a') as f:
            for doc in llama_parse_documents:
                content = getattr(doc, 'text', '')
                f.write(content + '\n')
        
        loader = UnstructuredMarkdownLoader(output_file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embed_model = HuggingFaceEmbeddings(model_name="bert-base-uncased")
        persist_directory = "chroma_db_llamaparse1"
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name="rag"
        )
        
        return vs, embed_model

    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise

def main():
    try:
        vs, embed_model = create_vector_database()

        chat_model = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        vectorstore = Chroma(
            embedding_function=embed_model,
            persist_directory="chroma_db_llamaparse1",
            collection_name="rag"
        )

        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        def set_custom_prompt():
            prompt = PromptTemplate(template=custom_prompt_template,
                                    input_variables=['context', 'question'])
            return prompt

        prompt = set_custom_prompt()
        
        qa = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )

        st.title("Document Query System")

        query = st.text_input("Enter your query:")
        
        if st.button("Get Answer"):
            if query:
                response = qa.invoke({"query": query})
                st.write("**Answer:**")
                st.write(response)
            else:
                st.error("Please enter a query.")

    except Exception as e:
        st.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
