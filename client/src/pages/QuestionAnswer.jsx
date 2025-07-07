import { useState } from "react";
import { Loader2, SendHorizonal, MessageSquareQuote } from "lucide-react";
import axios from "axios";

function QuestionAnswer() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const formData = new FormData();
      formData.append("question", question);

      const res = await axios.post("http://localhost:5000/question-answering", formData);
      setResponse(res.data.response);
    } catch (err) {
      setResponse({ error: "Failed to get an answer. Please try again later." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold text-purple-700 mb-8 flex items-center gap-2">
        <MessageSquareQuote className="w-8 h-8" />
        Ask a Question
      </h1>

      <form onSubmit={handleQuestionSubmit} className="flex flex-col sm:flex-row gap-4 mb-8">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Type your question here..."
          className="flex-1 p-4 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-500"
        />
        <button
          type="submit"
          className="bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-lg flex items-center justify-center"
          disabled={loading}
        >
          {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <SendHorizonal className="w-5 h-5" />}
        </button>
      </form>

      {response && response.error && (
        <div className="bg-red-100 text-red-700 p-4 rounded-lg">{response.error}</div>
      )}

      {response && !response.error && (
        <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-6 space-y-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-800">✅ Answer</h2>
            <p className="mt-2 text-gray-700">{response.answer}</p>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-800">🔍 Confidence</h2>
            <p className="mt-2 text-gray-700">{response.confidence}</p>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-800">📚 Source Context</h2>
            <p className="mt-2 text-gray-700 whitespace-pre-line">{response.source_context}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export default QuestionAnswer;
