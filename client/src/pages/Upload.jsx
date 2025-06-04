import { useContext, useState } from 'react';
import axios from 'axios';
import { AppContext } from '../context/AppContext';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, AlertCircle, UploadCloud } from 'lucide-react';

function Upload() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState(null);
  const [error, setError] = useState(null);
  const [generateSuccessful, setGenerateSuccessful] = useState(false);

  const { setSummaryData } = useContext(AppContext);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('⚠️ Please select a file before submitting.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      setMessage(null);
      setError(null);

      const response = await axios.post('http://localhost:5000/generate-summary-mcqs', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSummaryData(response.data);
      setGenerateSuccessful(true);
      setMessage('✅ Summary and MCQs generated successfully!');
    } catch (err) {
      console.error(err);
      setError('❌ Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-12 bg-white p-10 rounded-2xl shadow-xl border border-gray-100">
      <h2 className="text-4xl font-bold text-center text-purple-700 mb-8">
        📄 Upload Your PDF
      </h2>

      {message && (
        <div className="flex items-center gap-2 bg-green-100 text-green-800 px-4 py-3 rounded mb-4 border border-green-300">
          <CheckCircle className="w-5 h-5" />
          <span>{message}</span>
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 bg-red-100 text-red-800 px-4 py-3 rounded mb-4 border border-red-300">
          <AlertCircle className="w-5 h-5" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        <label
          htmlFor="file-upload"
          className="flex flex-col items-center justify-center w-full h-48 border-2 border-dashed border-purple-300 rounded-lg cursor-pointer bg-gray-50 hover:bg-purple-50 transition"
        >
          <div className="flex flex-col items-center justify-center pt-5 pb-6">
            <UploadCloud className="w-10 h-10 mb-3 text-purple-400" />
            <p className="mb-1 text-sm text-gray-500">Click to upload or drag your PDF here</p>
            <p className="text-xs text-gray-400">Only PDF files are supported</p>
          </div>
          <input
            id="file-upload"
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>

        {file && (
          <div className="text-sm text-gray-700 border-t pt-4">
            📎 <strong>{file.name}</strong> ({(file.size / 1024).toFixed(2)} KB)
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className={`w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-lg font-semibold transition ${
            loading ? 'opacity-50 cursor-not-allowed' : ''
          }`}
        >
          {loading ? 'Generating...' : 'Generate Summary & MCQs'}
        </button>

        {generateSuccessful && (
          <div className="flex flex-col sm:flex-row gap-4 mt-4">
            <button
              type="button"
              onClick={() => navigate('/summary-mcqs')}
              className="flex-1 bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg font-semibold transition"
            >
              🎯 View Summary & MCQs
            </button>
            <button
              type="button"
              onClick={() => navigate('/question-answering')}
              className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white py-3 px-6 rounded-lg font-semibold transition"
            >
              💬 Ask Questions
            </button>
          </div>
        )}
      </form>
    </div>
  );
}

export default Upload;
