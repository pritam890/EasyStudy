import { Link } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="bg-purple-700 text-white p-4 shadow-lg flex justify-between items-center">
      <h1 className="text-2xl font-bold">EasyStudy</h1>
      <div className="space-x-4">
        <Link to="/" className="hover:underline">Home</Link>
        <Link to="/upload" className="hover:underline">Generate</Link>
        <Link to="/summary-mcqs" className="hover:underline">Result</Link>
        <Link to="/question-answering" className="hover:underline">Q&A</Link>
        <Link to="/tutorials" className="hover:underline">Tutorials</Link>
      </div>
    </nav>
  );
}

export default Navbar;
