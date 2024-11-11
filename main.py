from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load a text document; replace 'your_file.txt' with the path to your document.
loader = TextLoader("lily.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Initialize the OpenAI embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs, embedding)

# Initialize the language model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"max_length": 512,
                  "min_length": 25,
                  "temperature": 0.6
                  }
)

# Create a Retrieval-based QA chain
qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_store.as_retriever())

query = "Describe Lily Of The Valley in detail"
answer = qa_chain.run(query)

print(answer)
