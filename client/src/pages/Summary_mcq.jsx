// pages/Summary_mcq.jsx
import { useContext } from 'react';
import { AppContext } from '../context/AppContext';
import { CheckCircle, AlertCircle } from 'lucide-react';

function Summary_mcq() {
  const { summaryData } = useContext(AppContext);

  if (!summaryData) {
    return (
      <div className="flex justify-center items-center mt-20">
        <div className="flex items-center gap-2 text-red-600 text-lg">
          <AlertCircle className="w-6 h-6" />
          <span>No data to display. Please upload a PDF first.</span>
        </div>
      </div>
    );
  }

  const { summary, mcqs } = summaryData;

  return (
    <div className="max-w-4xl mx-auto p-8 mt-12 bg-white rounded-2xl shadow-xl border border-gray-200">
      <div className="mb-12">
        <h2 className="text-4xl font-extrabold text-purple-700 mb-4">📝 Summary</h2>
        <p className="text-lg text-gray-700 whitespace-pre-line leading-relaxed">{summary}</p>
      </div>

      <div>
        <h2 className="text-4xl font-extrabold text-purple-700 mb-8">❓ Multiple Choice Questions</h2>
        <ol className="space-y-8 list-decimal list-inside text-gray-800">
          {mcqs.map((item, index) => (
            <li key={index} className="bg-gray-50 p-6 rounded-lg shadow-sm hover:shadow-md transition-all duration-200">
              <p className="font-semibold text-lg">{item.question}</p>
              <ul className="list-disc pl-6 mt-2 space-y-1">
                {item.options.map((opt, i) => (
                  <li key={i} className="text-gray-700">{opt}</li>
                ))}
              </ul>
              <div className="flex items-center text-green-600 mt-3 text-sm">
                <CheckCircle className="w-4 h-4 mr-1" />
                <span>Correct Answer: {item.answer}</span>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export default Summary_mcq;
