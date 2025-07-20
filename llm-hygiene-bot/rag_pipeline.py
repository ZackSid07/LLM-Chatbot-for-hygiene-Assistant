import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def load_docs(folder_path="documents"):
    file_path = os.path.join(folder_path, "hygiene.txt")
    loader = TextLoader(file_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

def create_vectorstore():
    documents = load_docs()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents, embeddings, persist_directory="hygiene_vector_db")
    vectordb.persist()
    return vectordb

def get_qa_chain():
    vectordb = Chroma(persist_directory="hygiene_vector_db", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-4o")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
