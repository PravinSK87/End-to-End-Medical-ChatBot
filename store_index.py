from langchain.vectorstores import Chroma
from src.helper import load_pdf, text_splitter,download_hfembeddings
from dotenv import load_dotenv
import os
load_dotenv()

##Data Ingestion
extracted_data = load_pdf("Data/")

## Text splitteer and chunking
text_chunks = text_splitter(extracted_data)

##Embeddings for vector store
embeddings = download_hfembeddings()

##Vector store
vector_db = Chroma.from_documents(text_chunks,embeddings,persist_directory="db")

