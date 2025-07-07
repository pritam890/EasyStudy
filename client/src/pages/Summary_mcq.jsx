import { useContext, useState, useEffect } from 'react';
import { AppContext } from '../context/AppContext';
import { AlertCircle, BookOpenCheck, Award } from 'lucide-react';

function Summary_mcq() {
  const { summary, qaList } = useContext(AppContext);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [score, setScore] = useState(0);

  useEffect(() => {
    // Calculate score whenever an answer is selected
    let correctCount = 0;
    qaList?.forEach((qa, index) => {
      if (selectedAnswers[index] === qa.answer) {
        correctCount++;
      }
    });
    setScore(correctCount);
  }, [selectedAnswers, qaList]);

  const handleAnswerClick = (qIndex, option) => {
    if (selectedAnswers[qIndex]) return; // Prevent changing answers
    setSelectedAnswers(prev => ({ ...prev, [qIndex]: option }));
  };

  if (!summary) {
    return (
      <div className="flex justify-center items-center mt-20">
        <div className="flex items-center gap-2 text-red-600 text-lg">
          <AlertCircle className="w-6 h-6" />
          <span>No data to display. Please upload a PDF first.</span>
        </div>
      </div>
    );
  }

  const allAnswered = qaList?.length > 0 && Object.keys(selectedAnswers).length === qaList.length;

  return (
    <div className="max-w-5xl mx-auto p-8 mt-12 bg-white rounded-3xl shadow-2xl border border-gray-100 space-y-12">
      {/* Summary Section */}
      <section>
        <h2 className="text-4xl font-extrabold text-purple-700 flex items-center gap-2 mb-4">
          <BookOpenCheck className="w-8 h-8 text-purple-500" />
          Summary
        </h2>
        <p className="text-lg text-gray-800 leading-relaxed tracking-wide whitespace-pre-line bg-purple-50 p-6 rounded-xl shadow-inner">
          {summary}
        </p>
      </section>

      {/* MCQ Section */}
      {qaList?.length > 0 && (
        <section>
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-3xl font-bold text-gray-800">🧠 Practice MCQs</h3>
            {allAnswered && (
              <div className="flex items-center gap-2 text-green-700 text-xl font-semibold">
                <Award className="w-6 h-6" />
                Score: {score} / {qaList.length}
              </div>
            )}
          </div>

          <div className="space-y-6">
            {qaList.map((qa, index) => {
              const userAnswer = selectedAnswers[index];
              const isCorrect = userAnswer === qa.answer;

              return (
                <div
                  key={index}
                  className="p-6 bg-white border border-gray-200 rounded-2xl shadow-md hover:shadow-lg transition-shadow"
                >
                  <h4 className="text-xl font-semibold text-indigo-700 mb-4">
                    Q{index + 1}. {qa.question}
                  </h4>
                  <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2">
                    {qa.options.map((option, optIndex) => {
                      const isSelected = userAnswer === option;
                      const isAnswer = qa.answer === option;

                      return (
                        <li
                          key={optIndex}
                          onClick={() => handleAnswerClick(index, option)}
                          className={`
                            px-4 py-2 rounded-lg cursor-pointer border text-gray-800 transition-all duration-200
                            ${
                              userAnswer
                                ? isSelected
                                  ? isCorrect
                                    ? 'bg-green-100 border-green-400 font-semibold'
                                    : 'bg-red-100 border-red-400 text-red-600'
                                  : isAnswer
                                  ? 'bg-green-50 border-green-300'
                                  : 'border-gray-200'
                                : 'hover:bg-indigo-50 border-gray-300'
                            }
                          `}
                        >
                          {option}
                        </li>
                      );
                    })}
                  </ul>

                  {userAnswer && (
                    <div className="mt-4 text-sm">
                      {isCorrect ? (
                        <p className="text-green-600 font-semibold">✅ Correct!</p>
                      ) : (
                        <p className="text-red-600 font-semibold">
                          ❌ Incorrect. Correct Answer: <span className="text-green-700">{qa.answer}</span>
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

export default Summary_mcq;
