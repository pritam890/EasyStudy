import React, { useContext } from 'react';
import { assets } from '../assets/assets';
import { Link, useNavigate } from 'react-router-dom';
import { AppContext } from '../context/AppContext';

const Navbar = () => {
  const navigate = useNavigate();
  const {isGenerated,results} = useContext(AppContext);

  return (
    <div className="flex items-center justify-between px-6 py-4 ">
      
      {/* Logo */}
      <Link to="/">
        <img
          src={assets.my_logo2}
          alt="Logo"
          className="w-24 sm:w-32 lg:w-40 object-contain"
        />
      </Link>

      {/* Navigation Buttons */}
      <div className="flex items-center gap-3 sm:gap-6">
        <Link
          to="/"
          className="text-gray-600 hover:text-indigo-600 font-medium text-sm sm:text-base"
        >
          Home
        </Link>
        
        {isGenerated && <button
          onClick={() => navigate('/summary-mcqs')}
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-full text-sm font-semibold transition duration-300"
        >
          Summary & MCQ
        </button>}
        {isGenerated && <button
          onClick={() => navigate('/question-answering')}
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-full text-sm font-semibold transition duration-300"
        >
          Q&A
        </button>}
        {isGenerated && <button
          onClick={() => navigate('/tutorials')}
          className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-full text-sm font-semibold transition duration-300"
        >
          Tutorials
        </button>}
      </div>
      
    </div>
  );
};

export default Navbar;