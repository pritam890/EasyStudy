import os
import traceback
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = "data"

# LangChain setup
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192", temperature=0.5)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pydantic model for structured output
class QAResponse(BaseModel):
    answer: str = Field(description="The final answer to the question.")
    confidence: str = Field(description="Confidence level: Low, Medium, or High.")
    source_context: str = Field(description="Relevant context or source info from the document.")

qa_parser = PydanticOutputParser(pydantic_object=QAResponse)

# Prompt template
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

# ---------- Q&A FUNCTION ----------
def answer_question(question: str):
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
        structured_response = qa_chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "success": True,
            "response": structured_response.dict()
        }

    except Exception as e:
        traceback.print_exc()
        # Fallback: try raw LLM response
        try:
            raw_response = (qa_prompt | llm).invoke({
                "context": context,
                "question": question
            })
            return {
                "success": False,
                "error": "Parsing failed",
                "raw_response": raw_response,
                "details": str(e)
            }
        except Exception as fallback_error:
            return {
                "success": False,
                "error": "Failed to run fallback response",
                "details": str(fallback_error)
            }
