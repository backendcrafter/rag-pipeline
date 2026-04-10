from langchain_groq import ChatGroq
import requests
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

load_dotenv()
MY_API=os.getenv("MY_API")

vector_store=Chroma(collection_name="first_collaction",embedding_function=embeddings,persist_directory="./chroma_langchain_DB")
llm=ChatGroq(model="llama-3.3-70b-versatile",api_key=MY_API)

while True:
    query=input("enter query or 'exit' to quit : ")
    if(query=="exit"):
        break
    else:
        results=vector_store.similarity_search(query,k=2)
        RAG_answers=""
        for r in results:
            RAG_answers+=r.page_content+" / "

        input_prompt="we are performing a RAG search so i will give you my query and its 3 top answers you have to answer me acconding to that\n"
        input_prompt+="query : "+query+"\n"
        input_prompt+="answers : "+RAG_answers+"\n"

        response=llm.invoke([HumanMessage(content=input_prompt)])
        print(response.content)