import os
import tkinter as tk
from tkinter import filedialog
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Set up the Hugging Face API token
load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_key:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Initialize Tkinter and hide the root window
root = tk.Tk()
root.withdraw()  # Hide the root window as we just need the file dialog

# Ask the user to select a document manually
file_path = filedialog.askopenfilename(
    title="Select a Document",
    filetypes=(("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*"))
)

# Ensure a file was selected
if file_path:
    print(f"Document selected: {file_path}")

    # Load the selected document
    loader = TextLoader(file_path) if file_path.endswith('.txt') else PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10, separators=['\n\n', '\n', '.', ''])
    splits = text_splitter.split_documents(documents)

    # Initialize the embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    vector_store = Chroma.from_documents(splits, embedding)

    # Initialize the language model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={
            "max_length": 512,
            "min_length": 16,
            "temperature": 0.25,
            "num_beams": 3,
            "repetition_penalty": 1.1
        }
    )

    # Create the Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Start the query loop
    query = input("Enter query: ")
    while query.lower() != 'bye':
        answer = qa_chain.invoke(query)['result']
        print(answer)
        query = input("Enter query: ")

else:
    print("No file selected. Exiting.")
