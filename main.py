from dotenv import load_dotenv
import os
import gradio as gr
from google import genai
from google.genai import types
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

MODEL = "gemini-3.6-flash"
DB_NAME = "vector_db"

gemini_key = os.getenv("gemini_api_key")

if not gemini_key:
    print("no api was found")
else:
    print("api found")

client = genai.Client(api_key=gemini_key)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

SYSTEM_PROMPT_TEMPLATE = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""


def answer_question(question: str, history):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt
        )
    )

    return response.text
gr.ChatInterface(answer_question).launch()