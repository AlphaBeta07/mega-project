import { useState, useRef, useEffect } from 'react';
import AddSourceModal from './AddSourceModal';
import {
  Share, Settings, Grid, Search, Plus, X,
  ChevronDown, FileText, FileAudio, FileVideo, FileBarChart,
  BrainCircuit, Layers, MessageSquare, Sparkles,
  ArrowRight, MoreVertical, PanelLeft, PanelRight,
  Table, Network, PlaySquare, PenTool, Loader2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';
function App() {
  const [sources, setSources] = useState<any[]>([]);
  const [selectedSourceIds, setSelectedSourceIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  
  const backendUrl = "http://localhost:8000";

  useEffect(() => {
    fetchSources();
  }, []);

  const fetchSources = async () => {
    try {
      const res = await fetch(`${backendUrl}/api/sources`);
      const data = await res.json();
      if (data.sources) {
        setSources(data.sources);
        setSelectedSourceIds(new Set(data.sources.map((s: any) => s.id)));
      }
    } catch (err) {
      console.error("Failed to fetch sources", err);
    }
  };

  const handleRemoveSource = async (id: string) => {
    try {
      await fetch(`${backendUrl}/api/sources/${id}`, { method: 'DELETE' });
      setSources(sources.filter(s => s.id !== id));
    } catch (err) {
      console.error("Failed to delete source", err);
    }
  };

  const handleUploadSuccess = (newSources: any[]) => {
    setSources(prev => {
      const existingIds = new Set(prev.map(s => s.id));
      const filteredNew = newSources.filter(s => !existingIds.has(s.id));
      
      const newSourceIds = filteredNew.map(s => s.id);
      setSelectedSourceIds(current => new Set([...current, ...newSourceIds]));
      
      return [...prev, ...filteredNew];
    });
  };

  const handleSendMessage = async (overrideText?: string) => {
    const textToSend = typeof overrideText === 'string' ? overrideText : inputValue;
    if (!textToSend.trim()) return;

    const userMessage = { role: "user", content: textToSend };
    setMessages(prev => [...prev, userMessage]);
    if (typeof overrideText !== 'string') {
      setInputValue("");
    }
    setIsChatting(true);

    try {
      const res = await fetch(`${backendUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages,
          selected_source_ids: Array.from(selectedSourceIds)
        }),
      });
      const data = await res.json();
      
      setMessages(prev => [...prev, { role: "assistant", content: data.response }]);
    } catch (err) {
      console.error("Chat error", err);
      setMessages(prev => [...prev, { role: "assistant", content: "Sorry, I encountered an error. Please ensure LM Studio is running on port 1234 and the backend is running." }]);
    } finally {
      setIsChatting(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo-container">
            <Layers size={18} fill="currentColor" />
          </div>
          <span className="notebook-title">StudySnap AI</span>
          <button className="btn-create-notebook">
            <Plus size={16} />
            Create notebook
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-content">
        
        {/* Left Sidebar: Sources */}
        <aside className="panel sidebar-left">
          <div className="panel-header">
            <div className="panel-title">
              Sources
            </div>
            <button className="icon-btn">
              <PanelLeft size={18} />
            </button>
          </div>
          
          <div className="sources-content">
            <button className="btn-add-source" onClick={() => setIsModalOpen(true)}>
              <Plus size={16} />
              Add sources
            </button>
            
            <div className="search-section">
              <span className="search-title">Search the web for new sources</span>
              <div className="search-bar">
                <button className="dropdown-btn"><Search size={14} /> Web <ChevronDown size={14} /></button>
                <button className="dropdown-btn"><Sparkles size={14} /> Fast Research <ChevronDown size={14} /></button>
                <button className="search-input-wrapper"><Search size={14} /></button>
              </div>
            </div>
            
            {sources.length === 0 ? (
              <div className="empty-sources">
                <FileText size={32} className="empty-sources-icon" />
                <div className="empty-sources-title">Saved sources will appear here</div>
                <div className="empty-sources-desc">
                  Click Add source above to add PDFs, websites, text, videos, or audio files. Or import a file directly from Google Drive.
                </div>
              </div>
            ) : (
              <div style={{display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px'}}>
                {sources.map(src => (
                  <div key={src.id} style={{padding: '12px', backgroundColor: 'var(--bg-button)', borderRadius: '8px', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: selectedSourceIds.has(src.id) ? '1px solid var(--accent-color)' : '1px solid transparent', opacity: selectedSourceIds.has(src.id) ? 1 : 0.6}}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden'}}>
                      <input 
                        type="checkbox" 
                        checked={selectedSourceIds.has(src.id)}
                        onChange={(e) => {
                          const newSet = new Set(selectedSourceIds);
                          if (e.target.checked) newSet.add(src.id);
                          else newSet.delete(src.id);
                          setSelectedSourceIds(newSet);
                        }}
                        style={{accentColor: 'var(--accent-color)', width: '16px', height: '16px', cursor: 'pointer', flexShrink: 0}}
                      />
                      <FileText size={16} color="var(--accent-color)" style={{flexShrink: 0}} />
                      <span style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}} title={src.filename}>{src.filename}</span>
                    </div>
                    <button onClick={() => handleRemoveSource(src.id)} className="icon-btn" style={{width: '24px', height: '24px', flexShrink: 0}} title="Remove Source">
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* Center Panel: Chat */}
        <section className="panel chat-center">
          <div className="panel-header" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
            <div className="panel-title">Chat</div>
            <div style={{display: 'flex', gap: '8px', alignItems: 'center'}}>
              {messages.length > 0 && (
                <button 
                  onClick={() => setMessages([])} 
                  className="btn-add-source"
                  style={{padding: '4px 12px', height: '28px', fontSize: '13px'}}
                >
                  <Plus size={14} /> New Chat
                </button>
              )}
              <button className="icon-btn"><MoreVertical size={18} /></button>
            </div>
          </div>
          
          <div className="chat-content" style={{justifyContent: messages.length > 0 ? 'flex-start' : 'center', overflowY: 'auto', paddingBottom: '100px'}}>
            {messages.length === 0 ? (
              <>
                <div className="welcome-icon">👋</div>
                <h1 className="welcome-title">Let's start studying...</h1>
                <p className="welcome-desc">This is your blank canvas to understand, create, or make progress on something new. I can help you get started or you can go ahead and add your own sources.</p>
                <h2 className="suggestions-title">What would you like this notebook to help you do?</h2>
                <div className="suggestions-list">
                  {/* <button className="suggestion-btn" onClick={() => handleSendMessage("Start a project")}>Start a project</button> */}
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the following audio transcript into structured educational notes using Markdown. You are an expert educational assistant that creates clear, structured study notes. You MUST include all five sections with these exact headings: # Title, ## Key Points, ## Explanation, ## Examples, ## Summary in detailed.")}>Audio to notes</button>
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the content from the provided YouTube link/video into structured educational notes using Markdown. You are an expert educational assistant that creates clear, structured study notes. You MUST include all five sections with these exact headings: # Title, ## Key Points, ## Explanation, ## Examples, ## Summary, in detailed format.")}>YouTube to notes</button>
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Learn or understand something")}>Learn or understand something</button>
                  {/* <button className="suggestion-btn" onClick={() => handleSendMessage("Create a podcast, video, slide deck, etc.")}>Create a podcast, video, slide deck, etc.</button> */}
                </div>
              </>
            ) : (
              <div style={{width: '100%', display: 'flex', flexDirection: 'column', gap: '24px'}}>
                {messages.map((msg, i) => (
                  <div key={i} style={{alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%', backgroundColor: msg.role === 'user' ? 'var(--bg-button)' : 'transparent', padding: msg.role === 'user' ? '12px 16px' : '0', borderRadius: '16px', lineHeight: '1.6'}}>
                    {msg.role === 'assistant' && <div style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--accent-color)'}}><Sparkles size={16} /> <span>StudySnap AI</span></div>}
                    <div className={msg.role === 'assistant' ? 'markdown-body' : ''}>
                      {msg.role === 'assistant' ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                ))}
                {isChatting && (
                  <div style={{display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-color)'}}>
                    <Sparkles size={16} /> <Loader2 size={16} className="animate-spin" /> Thinking...
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className="chat-input-container">
            <input 
              type="text" 
              className="chat-input" 
              placeholder="Ask a question or create something" 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <span className="input-sources-count">{sources.length} sources</span>
            <button className="btn-send" onClick={handleSendMessage} disabled={isChatting || !inputValue.trim()}>
              <ArrowRight size={16} />
            </button>
          </div>
          <div className="disclaimer">StudySnap AI can be inaccurate; please double check its responses.</div>
        </section>

        {/* Right Sidebar: Studio */}
        <aside className="panel sidebar-right">
          <div className="panel-header">
            <div className="panel-title">Studio</div>
            <button className="icon-btn"><PanelRight size={18} /></button>
          </div>
          
          <div className="studio-content">
            <div className="language-banner">
              Create an Audio Overview in: हिन्दी, বাংলা, ગુજરાતી, ಕನ್ನಡ, മലയാളം, മറാഠി, ਪੰਜਾਬੀ, தமிழ், తెలుగు
            </div>
            
            <div className="studio-grid">
              <div className="studio-card card-audio"><div className="studio-card-left"><FileAudio size={18} className="studio-card-icon" /><span className="studio-card-title">Audio Overview</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-slide"><div className="studio-card-left"><PlaySquare size={18} className="studio-card-icon" style={{color: '#d4e6ba'}} /><span className="studio-card-title">Slide Deck</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-video"><div className="studio-card-left"><FileVideo size={18} className="studio-card-icon" style={{color: '#b6e2c3'}} /><span className="studio-card-title">Video Overview</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-mindmap"><div className="studio-card-left"><Network size={18} className="studio-card-icon" style={{color: '#e5b3d6'}} /><span className="studio-card-title">Mind Map</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-reports"><div className="studio-card-left"><FileText size={18} className="studio-card-icon" style={{color: '#dfc98a'}} /><span className="studio-card-title">Reports</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-flashcards"><div className="studio-card-left"><BrainCircuit size={18} className="studio-card-icon" style={{color: '#e5b0a3'}} /><span className="studio-card-title">Flashcards</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-quiz"><div className="studio-card-left"><MessageSquare size={18} className="studio-card-icon" style={{color: '#a3d8d3'}} /><span className="studio-card-title">Quiz</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-infographic"><div className="studio-card-left"><FileBarChart size={18} className="studio-card-icon" style={{color: '#d6b3e5'}} /><span className="studio-card-title">Infographic</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
              <div className="studio-card card-table"><div className="studio-card-left"><Table size={18} className="studio-card-icon" style={{color: '#b3bfe5'}} /><span className="studio-card-title">Data Table</span></div><ChevronDown size={16} className="studio-card-arrow" style={{transform: 'rotate(-90deg)'}} /></div>
            </div>
            
            <div className="studio-empty">
              <PenTool size={24} />
              <div>
                <div style={{fontWeight: 500, color: 'var(--text-primary)', marginBottom: 4}}>Studio output will be saved here.</div>
                <div style={{fontSize: 13}}>After adding sources, you can create an Audio Overview, Study Guide, Briefing Doc, and more!</div>
              </div>
            </div>
          </div>
          
          <button className="btn-add-note">
            <Plus size={18} /> Add note
          </button>
        </aside>
        
      </main>

      <AddSourceModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        onUploadSuccess={handleUploadSuccess} 
        backendUrl={backendUrl} 
      />
    </div>
  );
}

export default App;
