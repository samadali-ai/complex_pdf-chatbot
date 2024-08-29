import os
import joblib
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from groq import Groq
from langchain_groq import ChatGroq
from parser import load_or_parse_data

# Load environment variables from .env file
load_dotenv()

def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using FastEmbedEmbeddings,
    and finally persists the embeddings into a Chroma vector database.
    """
    try:
        # Call the function to either load or parse the data
        print("Loading or parsing data...")
        llama_parse_documents = load_or_parse_data()
        
        # Debug: Check the type and length of llama_parse_documents
        print(f"Type of llama_parse_documents: {type(llama_parse_documents)}")
        print(f"Number of documents loaded: {len(llama_parse_documents)}")

        # Debugging: Print available attributes of the first document
        if len(llama_parse_documents) > 0:
            first_doc = llama_parse_documents[0]
            print(f"Attributes of the first document: {dir(first_doc)}")
            # If there is a method or attribute to get content
            print(f"First document content preview: {getattr(first_doc, 'text', 'No text attribute')}")

        # Write documents to a markdown file
        output_dir = 'data'
        output_file = os.path.join(output_dir, 'output.md')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        with open(output_file, 'a') as f:
            for doc in llama_parse_documents:
                content = getattr(doc, 'text', '')  # Adjust based on actual attribute or method
                f.write(content + '\n')
        
        # Debug: Confirm that the file was written
        print(f"Documents appended to file: {output_file}")
        
        # Load documents using UnstructuredMarkdownLoader
        print(f"Loading documents from path: {output_file}")
        loader = UnstructuredMarkdownLoader(output_file)
        documents = loader.load()
        
        # Debug: Check the number of documents loaded and their sample content
        print(f"Number of documents loaded from markdown: {len(documents)}")
        if len(documents) > 0:
            first_doc = documents[0]
            print(f"Attributes of the first loaded document: {dir(first_doc)}")
            print(f"First loaded document content preview: {getattr(first_doc, 'text', 'No text attribute')}")
        
        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # Debug: Check the number of chunks and sample chunk
        print(f"Total number of document chunks generated: {len(docs)}")
        if len(docs) > 0:
            first_chunk = docs[0]
            print(f"Attributes of the first document chunk: {dir(first_chunk)}")
            print(f"First document chunk content preview: {getattr(first_chunk, 'text', 'No text attribute')}")
        
        # Initialize Embeddings without model_name
        # You might need to check the correct way to initialize FastEmbedEmbeddings
        embed_model = FastEmbedEmbeddings()  # Adjust parameters if necessary
        
        # Debug: Confirm the embedding model initialization
        # Note: Adjust based on how you can get the model details if applicable
        print("Embedding model initialized.")
        
        # Create and persist a Chroma vector database from the chunked documents
        persist_directory = "chroma_db_llamaparse1"
        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=persist_directory,
            collection_name="rag"
        )
        
        # Debug: Confirm the persistence directory
        print(f"Vector database created and persisted in directory: {persist_directory}")
        
        # Final confirmation
        print('Vector DB created successfully!')
        
        return vs, embed_model

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def main():
    try:
        print("Starting the vector database creation process...")
        vs, embed_model = create_vector_database()
        print("Vector database creation process completed.")
        # Optionally, you can add more checks or validations here
    except Exception as e:
        print(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()

