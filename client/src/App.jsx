import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Summary_mcq from './pages/Summary_mcq';
import QuestionAnswering from './pages/QuestionAnswering';
import YouTube from './pages/Youtube';

function App() {
  return (
      <div className="flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/summary-mcqs" element={<Summary_mcq/>} />
            <Route path="/question-answering" element={<QuestionAnswering/>} />
            <Route path="/tutorials" element={<YouTube/>} />
          </Routes>
        </main>
        <Footer />
      </div>
  );
}

export default App;
