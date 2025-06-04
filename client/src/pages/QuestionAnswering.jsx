import React, { useState } from 'react';
import axios from 'axios';
import { Loader2, AlertCircle, Brain } from 'lucide-react';

function QuestionAnswering() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleQuestionSubmit = async (e) => {
    e.preventDefault();

    if (!question.trim()) {
      setError('⚠️ Please enter a question.');
      return;
    }

    setLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const formData = new FormData();
      formData.append('question', question);

      const response = await axios.post('http://localhost:5000/question-answering', formData);

      if (response.data.error) {
        setError(`❌ Server error: ${response.data.error}`);
        setAnswer(null);
      } else if (response.data.response) {
        setAnswer(response.data.response);
      } else {
        setError('❌ Unexpected server response.');
      }
    } catch (err) {
      if (err.response) {
        setError(`❌ Error ${err.response.status}: ${err.response.data.message || 'Server error'}`);
      } else {
        setError('❌ Failed to get answer. Please try again.');
      }
      console.error('Request failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto mt-12 p-8 bg-white rounded-2xl shadow-xl border border-gray-200">
      <h2 className="text-4xl font-bold text-purple-700 mb-6 text-center flex items-center justify-center gap-2">
        <Brain className="w-8 h-8" /> Ask a Question
      </h2>

      <form onSubmit={handleQuestionSubmit} className="space-y-6">
        <textarea
          rows="4"
          className="w-full p-4 text-lg border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
          placeholder="💬 Enter your question related to the document..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
        />

        <button
          type="submit"
          disabled={loading}
          className={`w-full flex justify-center items-center gap-2 bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-lg font-semibold transition duration-200 ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" /> Searching...
            </>
          ) : (
            '🔍 Get Answer'
          )}
        </button>
      </form>

      {error && (
        <div className="mt-6 flex items-center gap-2 bg-red-100 text-red-700 px-4 py-3 rounded-lg border border-red-300">
          <AlertCircle className="w-5 h-5" />
          <span>{error}</span>
        </div>
      )}

      {answer && (
        <div className="mt-8 p-6 bg-gray-50 border border-gray-200 rounded-xl shadow-sm transition-all">
          <h3 className="text-2xl font-semibold text-purple-600 mb-3 flex items-center gap-2">
            <Brain className="w-6 h-6" /> Answer
          </h3>
          <p className="text-gray-800 text-lg leading-relaxed mb-4">
            {typeof answer === 'string' ? answer : answer.answer}
          </p>

          {typeof answer === 'object' && (
            <div className="text-sm text-gray-600 space-y-1">
              {answer.confidence && (
                <div>
                  <strong>Confidence:</strong> {answer.confidence}
                </div>
              )}
              {answer.source_context && (
                <div>
                  <strong>Source:</strong> {answer.source_context}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default QuestionAnswering;
