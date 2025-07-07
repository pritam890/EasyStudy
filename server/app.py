from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader  # type: ignore
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import shutil
import requests

# Langchain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

# Custom modules
from mcq_generator import summarizer
from qa_engine import answer_question
from mcq_generator import get_mcq



app = Flask(__name__)
CORS(app)


VECTORSTORE_DIR = "data"

# Clear previous vector store
if os.path.exists(VECTORSTORE_DIR):
    shutil.rmtree(VECTORSTORE_DIR)

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Langchain setup
llm = ChatGroq(api_key=api_key, model="llama3-70b-8192", temperature=0.5)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

keyword_yt = ""
cached_videos = []

class genKeywords(BaseModel):
    keyword: str = Field(description="A concise 3 to 6 words extracted from the meaning of summary for searching related YouTube videos")

youtubeParser = PydanticOutputParser(pydantic_object=genKeywords)

YOUTUBE_KEYWORD_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template=("From the give summary of document extract the concise short 3 words keyword for youtube video searching.\n\n"
        "{format_instructions}\n\nText:\n{summary}"),
    partial_variables={"format_instructions": youtubeParser.get_format_instructions()}
)


def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from PDF file object."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Helper: Get YouTube Videos
def get_youtube_videos(query, max_results=5):
    try:
        search_url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            'part': 'snippet',
            'q': query,
            'type': 'video',
            'maxResults': max_results,
            'key': YOUTUBE_API_KEY
        }
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()

        videos = []
        for item in data.get('items', []):
            video_id = item['id']['videoId']
            snippet = item['snippet']
            videos.append({
                'title': snippet['title'],
                'video_id': video_id,
                'thumbnail': snippet['thumbnails']['medium']['url']
            })
        return videos
    except Exception as e:
        print("🔴 YouTube API Error:", e)
        return []

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running on http://localhost:5000"}), 200

# Route: Summarize
@app.route('/api/summarize_mcq', methods=['POST'])
def summarize_mcq():
    global cached_videos, keyword_yt
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400
    
    file = request.files['pdf']
    
    try:
        input_text = extract_text_from_pdf(file)

        if not input_text:
            return jsonify({"error": "No text provided"}), 400

        summary_text = summarizer(input_text)
        mcq_question = get_mcq(summary_text)

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        doc = Document(page_content=input_text)
        texts = text_splitter.split_documents([doc])

        vector_store = Chroma(
            collection_name='collection',
            embedding_function=embedding_model,
            persist_directory=VECTORSTORE_DIR
        )
        vector_store.add_documents(texts)

        # YouTube keyword generation
        keyword_chain = YOUTUBE_KEYWORD_PROMPT | llm | youtubeParser
        search_keyword = keyword_chain.invoke({'summary': summary_text})
        keyword_yt = search_keyword.keyword
        cached_videos = get_youtube_videos(keyword_yt, max_results=9)

        return jsonify({"success": True, "message": "Questions generated successfully","summary": summary_text, "mcq":mcq_question}),200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error generating summary: {str(e)}",
            "summary": []
        }), 500
    

@app.route('/question-answering', methods=['POST'])
def question_answering():
    question = request.form.get('question')
    
    if not question:
        return jsonify({"success": False, "error": "Question not provided"}), 400

    result = answer_question(question)

    if result.get("success"):
        return jsonify({"success": True, "response": result["response"]}), 200
    else:
        return jsonify({
            "success": False,
            "error": result.get("error", "Unknown error"),
            "details": result.get("details"),
            "raw_response": result.get("raw_response", None)
        }), 500


@app.route('/youtube', methods=['GET'])
def youtube_videos():
    return jsonify({
        "success": True,
        "videos": cached_videos
    }),200

# Run the app
if __name__ == '__main__':
    print("🚀 Server is running at http://localhost:5000")
    app.run(debug=True, port=5000)
