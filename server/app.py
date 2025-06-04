from flask import Flask, request, jsonify
import os
import traceback
from werkzeug.utils import secure_filename
import PyPDF2
from generator import get_mcq
from generator import summarizer as summary
from flask_cors import CORS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
import shutil
import requests

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
VECTORSTORE_DIR = "data"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# Global state
last_uploaded_text = ""
summary_text_data = ""
cached_videos = []
keyword_yt = ""

# Pydantic Models
class QAResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    confidence: str = Field(description="Confidence level: Low, Medium, or High.")
    source_context: str = Field(description="Relevant context or source info from the document.")

class genKeywords(BaseModel):
    keyword: str = Field(description="A concise 3 to 6 words extracted from the meaning of summary for searching related YouTube videos")

# Output parsers
qa_parser = PydanticOutputParser(pydantic_object=QAResponse)
youtubeParser = PydanticOutputParser(pydantic_object=genKeywords)

# Prompts
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an intelligent assistant that answers questions strictly using the provided context.\n"
        "Respond ONLY with a valid JSON in the following format — do not include any other text.\n\n"
        "{format_instructions}\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    ),
    partial_variables={"format_instructions": qa_parser.get_format_instructions()}
)

YOUTUBE_KEYWORD_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template=("From the give summary of document extract the concise short 3 words keyword for youtube video searching.\n\n"
        "{format_instructions}\n\nText:\n{summary}"),
    partial_variables={"format_instructions": youtubeParser.get_format_instructions()}
)

# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return "".join([page.extract_text() or "" for page in reader.pages])

# Helper: Get YouTube Videos
def get_youtube_videos(query, max_results=5):
    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': max_results,
        'key': YOUTUBE_API_KEY
    }
    response = requests.get(search_url, params=params)
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

# Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API running successfully"}), 200

@app.route('/generate-summary-mcqs', methods=['POST'])
def generate_summary_mcqs():
    global cached_videos, keyword_yt

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        text = extract_text_from_pdf(file_path)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        generated_summary = summary(text)
        mcq_list = get_mcq(text)

        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        texts = text_splitter.split_documents(docs)

        vector_store = Chroma(
            collection_name='collection',
            embedding_function=embedding_model,
            persist_directory=VECTORSTORE_DIR
        )
        vector_store.add_documents(texts)

        # YouTube keyword generation
        keyword_chain = YOUTUBE_KEYWORD_PROMPT | llm | youtubeParser
        search_keyword = keyword_chain.invoke({'summary': generated_summary})
        keyword_yt = search_keyword.keyword
        cached_videos = get_youtube_videos(keyword_yt, max_results=9)

        return jsonify({
            "summary": generated_summary,
            "mcqs": mcq_list
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/question-answering', methods=['POST'])
def question_answering():
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        vector_store = Chroma(
            collection_name='collection',
            embedding_function=embedding_model,
            persist_directory=VECTORSTORE_DIR
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        qa_chain = qa_prompt | llm | qa_parser
        try:
            structured_response = qa_chain.invoke({
                "context": context,
                "question": question
            })
            return jsonify({"response": structured_response.dict()})

        except Exception as parse_error:
            print("⚠️ Output parsing failed. Returning raw response.")
            raw_response = (qa_prompt | llm).invoke({
                "context": context,
                "question": question
            })
            return jsonify({
                "error": "Output parsing failed",
                "raw_response": raw_response,
                "details": str(parse_error)
            }), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process question: {str(e)}"}), 500

@app.route('/youtube', methods=['GET'])
def youtube_videos():
    return jsonify({
        "videos": cached_videos
    })

# Run app
if __name__ == "__main__":
    port = 5000
    print(f"App running on port {port}")
    app.run(debug=False, port=port)
