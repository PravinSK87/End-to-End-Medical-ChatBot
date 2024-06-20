from langchain.document_loaders import DirectoryLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings




def load_pdf(data):
    loader = DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


def text_splitter(extracted_data):
    txt_split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text = txt_split.split_documents(extracted_data)
    return text

def download_hfembeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings