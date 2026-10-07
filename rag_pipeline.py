import os
import glob
import tiktoken
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Setting up the enviorment
MODEL = "gemini-3.6-flash"
db_name = "vector_db"
load_dotenv()
gemini_key = os.getenv("gemini_api_key")

# Vaildting if I have my API working
if not gemini_key:
    print("no api was found")
else:
    print("api found")

# Defining my knowledge base path  
knowledge_base_path ="knowledge_base/**/*.md"
# Calculating how many files I have inside my KB
files = glob.glob(knowledge_base_path,recursive=True)
print(f"Found {len(files)} files in the knowledge base")

entire_kb = ""
# Calculating how many tokens do I have inside my kb
for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        entire_kb += f.read()
        entire_kb += "\n\n"

print(f"Total characters in knowledge base: {len(entire_kb):,}")

# Calculating how many tokens do Gemini model can comperhend
client = genai.Client(api_key=gemini_key)
response = client.models.count_tokens(
    model=MODEL,
    contents=entire_kb
)

print(f"Total tokens for {MODEL}: {response.total_tokens:,}")

#Load the documents 
folders = glob.glob("knowledge_base/*")
documents =[]

for folder in folders:
    doc_type =os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

#chunk the docs!
text_spltter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=150)
chunks = text_spltter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if os.path.exists(db_name):
    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=embeddings
    )
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_name
    )
    
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")   

collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


