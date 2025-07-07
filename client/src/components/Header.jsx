import React, { useState, useContext } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';
import { AppContext } from '../context/AppContext';
import { toast } from 'react-toastify';
import { useNavigate } from 'react-router-dom';

const Header = () => {
  const backendUrl = import.meta.env.VITE_BACKEND_URL
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfName, setPdfName] = useState('');
  const [showSubmit, setShowSubmit] = useState(false);
  const [showTest, setShowTest] = useState(false);
  const [processing, SetProcessing] = useState(false);
  const { setQAList, summary, setSummary, setIsGenerated} = useContext(AppContext);

  const navigate = useNavigate()

  // Triggered on file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setPdfFile(file);
    setPdfName(file.name);
    setShowTest(false);
    setShowSubmit(true);
  };

  // Triggered on submit button click
  const handlePDFUpload = async () => {
    if (!pdfFile) return;
    setShowSubmit(false)
    SetProcessing(true)
    const formData = new FormData();
    formData.append('pdf', pdfFile);

    try {
      const res = await axios.post(backendUrl+'/api/summarize_mcq', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = res.data;

      if (data.success) {
        setSummary(data.summary);
        console.log(data.summary);
        setQAList(data.mcq)
        console.log("\n")
        console.log(data.mcq)
        SetProcessing(false);
        setIsGenerated(true);
        setShowTest(true);
      } else {
        toast.error(`Server Error: ${data.message}`);
        SetProcessing(false);
      }
    } catch (error) {
      toast.error(`Error uploading PDF: ${error.message}`);
    }
  };

  return (
    <div className="min-h-[75vh] flex flex-col md:flex-row">
      {/* Left Section - Greeting */}
      <motion.div
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
        className="w-full md:w-1/2 flex flex-col justify-center items-start px-6 sm:px-12 lg:px-20 py-16"
      >
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-gray-800 leading-tight mb-4">
          Welcome to <br />
          <span className="text-orange-600">StudyHelp</span>
        </h1>
        <p className="text-md sm:text-lg text-gray-700 max-w-lg">
          Sharpen your preparation with our smart AI study assistant. Upload your PDF to get a summary, generated MCQs, Q&A, and tutorials related to your topic.
        </p>
      </motion.div>

      {/* Right Section - Upload */}
      <motion.div
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: 'easeOut', delay: 0.3 }}
        className="w-full md:w-1/2 flex justify-center items-center px-6 sm:px-12 py-16"
      >
        <div className="bg-white rounded-3xl border border-gray-200 shadow-xl hover:shadow-indigo-100 transition-all duration-300 p-8 sm:p-10 w-full max-w-md">
          <h2 className="text-3xl font-bold text-center text-orange-600 mb-6">
            📄 Upload PDF
          </h2>

          <div className="flex flex-col items-center gap-4">
            <label className="w-full text-center">
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                className="w-full cursor-pointer text-sm text-gray-700
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-600 file:text-white
                  hover:file:bg-indigo-700 transition-all"
              />
            </label>

            {pdfName && (
              <div className="text-sm text-green-600 font-medium text-center">
                ✅ Uploaded: <span className="font-semibold">{pdfName}</span>
              </div>
            )}

            {showSubmit && (
              <button
                onClick={handlePDFUpload}
                className="w-full mt-2 px-6 py-2 bg-indigo-600 text-white font-semibold rounded-full shadow hover:bg-indigo-700 transition-all duration-200"
              >
                🚀 Submit
              </button>
            )}

            {processing && (
              <button
                disabled
                className="w-full mt-2 px-6 py-2 bg-indigo-500 text-white font-semibold rounded-full opacity-60 cursor-wait"
              >
                ⏳ Generating Summary & MCQs...
              </button>
            )}

            {showTest && (
              <button
                onClick={() => navigate('/summary-mcqs')}
                className="w-full mt-4 px-6 py-2 bg-green-600 text-white font-semibold rounded-full shadow hover:bg-green-700 transition-all duration-200"
              >
                🎯 See Summary & MCQs
              </button>
            )}
          </div>
        </div>
      </motion.div>

    </div>
  );
};

export default Header;