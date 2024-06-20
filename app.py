from flask import Flask, render_template, jsonify,request
from src.helper import download_hfembeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from src.prompt import *
from store_index import *
import os

app = Flask(__name__)

load_dotenv()


embeddings = download_hfembeddings()

docsearch = vector_db

Prompt = PromptTemplate(template= Prompt_template,input_variables=["context", "question"])
chain_type_kwargs = {"prompt": Prompt}

llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={"max_new_tokens" : 512,
                            "temperature" : 0.8 })

qa = RetrievalQA.from_chain_type(llm=llm, 
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(search_kwargs={'k':2}), 
                                 chain_type_kwargs=chain_type_kwargs)

@app.route('/')
def index():
    return render_template('chat.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080,debug=True)