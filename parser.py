import os
import joblib
from llama_parse import LlamaParse

# Hardcode your API key here
llamaparse_api_key = 'llx-p72wg4Rg2QHYKP1RmWCHvhJUgtToZZjAAD4LE2YpVjRrPiJ5'

# Debugging: Check if the API key is set
print(f"Loaded API key: {llamaparse_api_key}")

def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"

    if os.path.exists(data_file):
        try:
            # Load the parsed data from the file
            print(f"Loading data from {data_file}...")
            parsed_data = joblib.load(data_file)
            print("Data loaded successfully.")
        except (EOFError, FileNotFoundError) as e:
            print(f"Error loading file: {e}")
            print("Re-parsing the document...")
            parsed_data = reparse_document(data_file)
    else:
        # File does not exist, parse the document
        print(f"{data_file} does not exist. Parsing the document...")
        parsed_data = reparse_document(data_file)

    print("Returning parsed data...")
    print(f"Parsed data: {parsed_data[:500]}")  # Print a snippet of the parsed data for checking
    return parsed_data

def reparse_document(data_file):
    # Perform the parsing step and store the result in llama_parse_documents
    parsingInstructionUber10k = """The provided document is a quarterly report filed by Uber Technologies,
    Inc. with the Securities and Exchange Commission (SEC).
    This form provides detailed financial information about the company's performance for a specific quarter.
    It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
    It contains many tables.
    Try to be precise while answering the questions"""
    
    # Ensure the API key is available
    if not llamaparse_api_key:
        raise ValueError("API key for LlamaParse is not set. Please check your code.")

    parser = LlamaParse(api_key=llamaparse_api_key,
                        result_type="markdown",
                        parsing_instruction=parsingInstructionUber10k,
                        max_timeout=5000)
    
    # Ensure the file path is correct and the file exists
    pdf_file_path = "/home/abdulsamad/llama_parse/2_of_1979_(e).pdf"
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"The PDF file at {pdf_file_path} does not exist.")
    
    print(f"Parsing document from {pdf_file_path}...")
    llama_parse_documents = parser.load_data(pdf_file_path)
    
    # Save the parsed data to a file
    print("Saving the parse results in .pkl format...")
    joblib.dump(llama_parse_documents, data_file)
    print("Data saved successfully.")

    return llama_parse_documents

# Example usage:
if __name__ == "__main__":
    print("Starting the data load or parse process...")
    load_or_parse_data()
