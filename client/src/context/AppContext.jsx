// context/AppContext.jsx
import { createContext, useState } from 'react';

export const AppContext = createContext();

const AppContextProvider = ({ children }) => {
  const [summaryData, setSummaryData] = useState(null);

  return (
    <AppContext.Provider value={{ summaryData, setSummaryData }}>
      {children}
    </AppContext.Provider>
  );
};
export default AppContextProvider
