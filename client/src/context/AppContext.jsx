import { createContext, useEffect, useState } from "react";
import { toast } from "react-toastify";
import axios from 'axios'
import { useNavigate } from "react-router-dom";

export const AppContext = createContext()


const AppContextProvider = (props)=>{
    const [isGenerated, setIsGenerated] = useState(false)
    const [qaList, setQAList] = useState(null)
    const [summary, setSummary] = useState('')
    const [results, setResults] = useState([])
    
    const value = {
        isGenerated, setIsGenerated, qaList, setQAList, results, setResults, summary, setSummary
    }

    return(
        <AppContext.Provider value={value}>
            {props.children}
        </AppContext.Provider>
    )
}
export default AppContextProvider