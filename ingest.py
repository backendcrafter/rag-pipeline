# load data, splite it, embaded it, store it in chroma

from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader=DirectoryLoader("docs/",glob="*.txt",loader_cls=TextLoader)
docs=loader.load()

text_spliter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=40)
splite=text_spliter.split_documents(docs)

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store=Chroma(
    collection_name="collection",
    embedding_function=embeddings,
    persist_directory="./docs_collections"
)

vector_store.add_documents(documents=splite)