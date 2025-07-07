import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Summary_mcq from './pages/Summary_mcq';
import QuestionAnswer from './pages/QuestionAnswer';
import YoutubeSection from './pages/YoutubeSection';
import { ToastContainer } from 'react-toastify'

function App() {
  return (
      <div className="px-4 sm:px-10 md:px-14 lg:px-28 min-h-screen bg-gradient-to-br from-white via-indigo-200 to-orange-100">
        <ToastContainer position='bottom-right'/>
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/summary-mcqs" element={<Summary_mcq/>} />
            <Route path="/question-answering" element={<QuestionAnswer/>} />
            <Route path="/tutorials" element={<YoutubeSection/>} />
          </Routes>
        </main>
        <Footer />
      </div>
  );
}

export default App;
