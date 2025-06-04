import os
import requests
from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Load environment variables
load_dotenv()

# API Keys
api_key = os.getenv("GROQ_API_KEY")

# Models and parsers
llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.5)
parser = StrOutputParser()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class QAResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    confidence: str = Field(description="Confidence level: Low, Medium, or High.")
    source_context: str = Field(description="Relevant context or source info from the document.")

qa_parser = PydanticOutputParser(pydantic_object=QAResponse)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an intelligent assistant helping users answer questions based on a document.\n\n"
        "{format_instructions}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    ),
    partial_variables={"format_instructions": qa_parser.get_format_instructions()}
)

# Global cache
last_uploaded_text = ""
summary_text_data = ""
cached_videos = []
keyword_yt = ""

