import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ThemeProvider } from './context/ThemeContext'
// import TokenTestPage from './TokenTestPage.tsx'  ← swap in to re-run token smoke test

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
    {/* <TokenTestPage /> */}
  </StrictMode>,
)
