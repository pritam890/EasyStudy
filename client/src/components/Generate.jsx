import { useNavigate } from 'react-router-dom';

function Generate() {
  const navigate = useNavigate();

  return (
    <div className="text-center mt-6">
      <button
        onClick={() => navigate('/upload')}
        className="bg-purple-600 hover:bg-purple-700 text-white font-semibold px-6 py-3 rounded-xl shadow-lg transition"
      >
        Generate Now
      </button>
    </div>
  );
}

export default Generate;
