import { useState, useEffect } from 'react';
import AddSourceModal from './AddSourceModal';
import AudioOverviewModal from './AudioOverviewModal';
import VideoOverviewModal from './VideoOverviewModal';
import InfographicModal from './InfographicModal';
import MindMapModal from './MindMapModal';
import MindMapRenderer from './MindMapRenderer';
import LandingPage from './components/LandingPage';
import StudentDashboard from './components/StudentDashboard';
import NoteReaderModal from './components/NoteReaderModal';
import AuthModal from './components/AuthModal';
import UserProfileModal from './components/UserProfileModal';
import BottomNav from './components/BottomNav';
import type { PublicNote } from './data/mockNotes';
import {
  Search, Plus, X, ArrowLeft,
  ChevronDown, FileText, FileAudio, FileVideo, FileBarChart, Network,
  BrainCircuit, MessageSquare, Sparkles,
  ArrowRight, MoreVertical, PanelLeft, PanelRight,
  PenTool, Loader2, Globe, TrendingUp, Headphones,
  ChevronRight, Maximize2, Trash2, Undo2, Redo2, Bold, Italic, Link, Code, Image, MoreHorizontal, FilePlus2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';
function App() {
  const [sources, setSources] = useState<any[]>([]);
  const [selectedSourceIds, setSelectedSourceIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<{ role: string, content: string }[]>([]);
  const [sessions, setSessions] = useState<{ id: string, title: string, messages: { role: string, content: string }[] }[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [searchLocation, setSearchLocation] = useState<'Web' | 'Drive'>('Web');
  const [researchMode, setResearchMode] = useState<'Fast Research' | 'Deep Research'>('Fast Research');
  const [isSearchExpanded, setIsSearchExpanded] = useState(false);
  const [searchBarQuery, setSearchBarQuery] = useState('');
  const [isLocationDropdownOpen, setIsLocationDropdownOpen] = useState(false);
  const [isResearchDropdownOpen, setIsResearchDropdownOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAudioModalOpen, setIsAudioModalOpen] = useState(false);
  const [isNoteMode, setIsNoteMode] = useState(false);
  const [noteTitle, setNoteTitle] = useState('New Note');
  const [noteContent, setNoteContent] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  const [audioOverviewUrl, setAudioOverviewUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false);
  const [videoOverviewUrl, setVideoOverviewUrl] = useState<string | null>(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [isInfographicModalOpen, setIsInfographicModalOpen] = useState(false);
  const [infographicMarkdown, setInfographicMarkdown] = useState<string | null>(null);
  const [isGeneratingInfographic, setIsGeneratingInfographic] = useState(false);

  const [isMindMapModalOpen, setIsMindMapModalOpen] = useState(false);
  const [mindMapData, setMindMapData] = useState<any>(null);
  const [isGeneratingMindMap, setIsGeneratingMindMap] = useState(false);

  const [responseLanguage, setResponseLanguage] = useState(() => localStorage.getItem('responseLanguage') || 'English');
  const [isResponseLangDropdownOpen, setIsResponseLangDropdownOpen] = useState(false);

  // 2-State System & Navigation States
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState<{ name: string; email: string; college: string } | null>({ name: 'Rohan Sharma', email: 'rohan.sharma@iitb.ac.in', college: 'IIT Bombay' });
  const [activeTab, setActiveTab] = useState<'home' | 'studio' | 'chat'>('home');
  const [selectedNoteForModal, setSelectedNoteForModal] = useState<PublicNote | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);

  useEffect(() => {
    localStorage.setItem('responseLanguage', responseLanguage);
  }, [responseLanguage]);

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
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);

    if (typeof overrideText !== 'string') {
      setInputValue("");
    }
    setIsChatting(true);

    let sessionIdToUse = currentSessionId;
    if (!sessionIdToUse) {
      sessionIdToUse = Date.now().toString();
      setCurrentSessionId(sessionIdToUse);
      setSessions(prev => [...prev, {
        id: sessionIdToUse!,
        title: textToSend.substring(0, 30) + (textToSend.length > 30 ? '...' : ''),
        messages: newMessages
      }]);
    } else {
      setSessions(prev => prev.map(s => s.id === sessionIdToUse ? { ...s, messages: newMessages } : s));
    }

    try {
      const res = await fetch(`${backendUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages,
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: responseLanguage
        }),
      });
      const data = await res.json();

      if (data.response) {
        const finalMessages = [...newMessages, { role: "assistant", content: data.response }];
        setMessages(finalMessages);
        setSessions(prev => prev.map(s => s.id === sessionIdToUse ? { ...s, messages: finalMessages } : s));
      }
    } catch (err) {
      console.error("Chat error", err);
      const errorMessages = [...newMessages, { role: "assistant", content: "Sorry, I encountered an error. Please ensure LM Studio is running on port 1234 and the backend is running." }];
      setMessages(errorMessages);
      setSessions(prev => prev.map(s => s.id === sessionIdToUse ? { ...s, messages: errorMessages } : s));
    } finally {
      setIsChatting(false);
    }
  };

  const handleSwitchSession = (id: string) => {
    if (id === 'new') {
      setMessages([]);
      setCurrentSessionId(null);
      return;
    }
    const session = sessions.find(s => s.id === id);
    if (session) {
      setCurrentSessionId(id);
      setMessages(session.messages);
    }
  };

  const handleGenerateAudioOverview = async () => {
    setIsGeneratingAudio(true);
    setAudioOverviewUrl(null);
    try {
      const res = await fetch(`${backendUrl}/api/audio-overview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: responseLanguage
        }),
      });
      const data = await res.json();
      if (data.success && data.audio_url) {
        setAudioOverviewUrl(data.audio_url);
      }
    } catch (err) {
      console.error("Failed to generate audio overview", err);
      alert("Failed to generate audio overview.");
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const handleGenerateVideoOverview = async (language: string, customPrompt: string) => {
    setIsGeneratingVideo(true);
    setVideoOverviewUrl(null);
    try {
      const res = await fetch(`${backendUrl}/api/video-overview`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          response_language: language,
          custom_prompt: customPrompt
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Server Error");
      if (data.success && data.video_url) {
        setVideoOverviewUrl(`${backendUrl}${data.video_url}`);
      }
    } catch (err: any) {
      console.error("Failed to generate video overview", err);
      alert("Failed to generate video overview: " + (err.message || "Unknown error"));
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  const handleGenerateInfographic = async (style: string, detailLevel: string, customPrompt: string) => {
    setIsGeneratingInfographic(true);
    setInfographicMarkdown(null);
    try {
      const res = await fetch(`${backendUrl}/api/infographic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          style,
          detail_level: detailLevel,
          custom_prompt: customPrompt,
          response_language: responseLanguage
        }),
      });
      const data = await res.json();
      if (data.success && data.markdown) {
        setInfographicMarkdown(data.markdown);
      }
    } catch (err) {
      console.error("Failed to generate infographic", err);
      alert("Failed to generate infographic.");
    } finally {
      setIsGeneratingInfographic(false);
    }
  };

  const handleGenerateMindMap = async (customPrompt: string) => {
    setIsGeneratingMindMap(true);
    setMindMapData(null);
    try {
      const res = await fetch(`${backendUrl}/api/mindmap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          custom_prompt: customPrompt,
          response_language: responseLanguage
        }),
      });
      const data = await res.json();
      if (data.success && data.mind_map) {
        setMindMapData(data.mind_map);
      }
    } catch (err) {
      console.error("Failed to generate mind map", err);
      alert("Failed to generate mind map.");
    } finally {
      setIsGeneratingMindMap(false);
    }
  };

  const handleNodeClick = (_nodeId: string, label: string, context: string) => {
    setIsLeftSidebarOpen(true);
    setInputValue(`Based on this context: "${context}", can you explain the concept of "${label}" in more detail?`);
  };

  const handleStudioAction = (type: string) => {
    let prompt = "";
    switch (type) {
      case 'slide': prompt = "Create ready-to-use presentation slides featuring AI-generated layouts and visual descriptions based on the selected sources. Format as 'Slide 1:', 'Visual:', 'Speaker Notes:', etc."; break;
      case 'mindmap': prompt = "Create a visual diagram that maps out and connects high-level concepts from the materials. Output a structured Markdown hierarchical list."; break;
      case 'reports': prompt = "Write a structured, written brief and executive summary of the selected content. Include an Executive Summary, Key Findings, and Conclusion."; break;
      case 'flashcards': prompt = "Create digital study flashcards extracting key terms and facts for quick memorization. Format as a Markdown table with 'Term' and 'Definition' columns."; break;
      case 'quiz': prompt = "Generate a multiple-choice quiz to test my comprehension of the source data. Provide 5 questions with 4 options each, and an answer key at the bottom."; break;
      case 'infographic': prompt = "Create a visual summary script incorporating icons, statistics, and graphic layouts based on the sources. Describe the layout, colors, and text for each section of the infographic."; break;
      case 'table': prompt = "Extract and organize key data points from the text into a structured, exportable Markdown table."; break;
      case 'pyq': prompt = "Analyze the provided previous year question papers. Use mathematical probability and pattern recognition to identify trends, frequency of topics, and weightage. Based on this analysis, predict the most likely questions and high-yield topics for the upcoming exam."; break;
      case 'audionotes': prompt = "Convert the provided audio transcript into structured educational notes using Markdown. You are an expert educational assistant that creates clear, structured study notes. You MUST include all five sections with these exact headings: # Title, ## Key Points, ## Explanation, ## Examples, ## Summary in detailed."; break;
    }
    if (prompt) {
      const finalPrompt = responseLanguage !== 'English'
        ? `${prompt}\n\nIMPORTANT: You MUST write your ENTIRE response in ${responseLanguage}. Translate any generated text, headers, and descriptions into ${responseLanguage}.`
        : prompt;
      handleSendMessage(finalPrompt);
    }
  };

  return (
    <div className="app-container">
      {/* Main Content Area */}
      <main className="main-content" style={{ position: 'relative', overflowY: activeTab === 'home' ? 'auto' : 'hidden' }}>
        {activeTab === 'home' ? (
          !isLoggedIn ? (
            <LandingPage
              onCreateNotebook={() => {
                if (isLoggedIn) {
                  setActiveTab('studio');
                  setIsModalOpen(true);
                } else {
                  setIsAuthModalOpen(true);
                }
              }}
              onSelectNote={(note) => setSelectedNoteForModal(note)}
            />
          ) : (
            <StudentDashboard
              user={currentUser}
              onOpenStudio={() => setActiveTab('studio')}
              onLogout={() => setIsLoggedIn(false)}
            />
          )
        ) : (
          <>
            {/* Top Bar for Studio View */}
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '42px',
              backgroundColor: 'var(--bg-panel)',
              borderBottom: '1px solid var(--border-color)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 16px',
              zIndex: 30
            }}>
              <button
                onClick={() => setActiveTab('home')}
                style={{
                  background: 'rgba(255, 255, 255, 0.08)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  color: '#F3F4F6',
                  padding: '4px 12px',
                  borderRadius: '9999px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px'
                }}
              >
                <ArrowLeft size={14} />
                Back to Student Dashboard
              </button>

              <div style={{ fontSize: '0.825rem', color: '#F59E0B', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Sparkles size={15} />
                AI Study Assistant & RAG Chatbot Studio
              </div>
            </div>

            {/* Right Sidebar Mini Rail (when closed) */}
            {!isRightSidebarOpen && (
          <aside className="panel sidebar-right-mini" style={{ width: '64px', minWidth: '64px', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', borderRight: '1px solid var(--border-color)', backgroundColor: 'var(--bg-panel)' }}>
            <button className="icon-btn" onClick={() => setIsRightSidebarOpen(true)} title="Open Studio" style={{ marginBottom: '16px' }}>
              <PanelRight size={18} />
            </button>
            <div style={{ width: '32px', height: '1px', backgroundColor: 'var(--border-color)', marginBottom: '16px', flexShrink: 0 }} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', overflowX: 'hidden', padding: '0 8px', scrollbarWidth: 'none', flex: 1, alignItems: 'center' }}>
              <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('pyq'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#332b26', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="PYQ Analysis">
                <TrendingUp size={18} style={{ color: '#ffcc99' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); setIsAudioModalOpen(true); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#2b2f3a', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Audio Overview">
                <FileAudio size={18} style={{ color: '#a8c7fa' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('audionotes'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#2f2b3a', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Audio to Notes">
                <Headphones size={18} style={{ color: '#c7a8fa' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              {/* <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('slide'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#353226', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Slide Deck">
                <PlaySquare size={18} style={{ color: '#d4e6ba' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button> */}
              <button onClick={() => { setIsRightSidebarOpen(true); setIsVideoModalOpen(true); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#2a332c', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Video Overview">
                <FileVideo size={18} style={{ color: '#b6e2c3' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              {/* <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('mindmap'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#352631', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Mind Map">
                <Network size={18} style={{ color: '#e5b3d6' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button> */}
              <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('reports'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#353326', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Reports">
                <FileText size={18} style={{ color: '#dfc98a' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('flashcards'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#352826', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Flashcards">
                <BrainCircuit size={18} style={{ color: '#e5b0a3' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('quiz'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#263336', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Quiz">
                <MessageSquare size={18} style={{ color: '#a3d8d3' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); setIsInfographicModalOpen(true); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#302635', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Infographic">
                <FileBarChart size={18} style={{ color: '#d6b3e5' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              <button onClick={() => { setIsRightSidebarOpen(true); setIsMindMapModalOpen(true); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#352c26', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Mind Map">
                <Network size={18} style={{ color: '#e5c9b3' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button>
              {/* <button onClick={() => { setIsRightSidebarOpen(true); handleStudioAction('table'); }} style={{ width: '40px', height: '40px', minHeight: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#2c2b36', cursor: 'pointer', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }} title="Data Table">
                <Table size={18} style={{ color: '#b3bfe5' }} />
                <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: '#e3e3e3' }} />
              </button> */}
            </div>

            <button onClick={() => { setIsRightSidebarOpen(true); setIsNoteMode(true); }} style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'white', color: 'black', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '16px', flexShrink: 0, boxShadow: '0 4px 12px rgba(0,0,0,0.2)', cursor: 'pointer' }} title="Add Note">
              <PenTool size={18} />
              <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'black' }} />
            </button>
          </aside>
        )}

        {/* Right Sidebar: Studio / Note */}
        {isRightSidebarOpen && (
          <aside className="panel sidebar-right" style={{ display: 'flex', flexDirection: 'column' }}>

            {isNoteMode ? (
              // NOTE EDITOR VIEW
              <>
                <div className="panel-header" style={{ padding: '12px 16px', borderBottom: 'none' }}>
                  <div style={{ display: 'flex', alignItems: 'center', fontSize: '13px', color: 'var(--text-secondary)', cursor: 'pointer' }} onClick={() => setIsNoteMode(false)}>
                    Studio <ChevronRight size={14} style={{ margin: '0 4px' }} /> <span style={{ color: 'var(--text-primary)' }}>Note</span>
                  </div>
                  <button className="icon-btn" title="Expand"><Maximize2 size={14} /></button>
                </div>

                <div style={{ padding: '0 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                  <input
                    type="text"
                    value={noteTitle}
                    onChange={e => setNoteTitle(e.target.value)}
                    style={{ fontSize: '20px', background: 'transparent', border: 'none', color: 'var(--text-primary)', outline: 'none', width: '100%' }}
                  />
                  <button className="icon-btn" onClick={() => { setNoteContent(''); setNoteTitle('New Note'); setIsNoteMode(false); }} title="Delete Note"><Trash2 size={18} /></button>
                </div>

                <div style={{ padding: '0 16px', display: 'flex', alignItems: 'center', gap: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '12px', color: 'var(--text-secondary)', overflowX: 'auto', flexWrap: 'nowrap', scrollbarWidth: 'none' }}>
                  <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                    <button className="icon-btn" style={{ padding: '4px' }}><Undo2 size={16} /></button>
                    <button className="icon-btn" style={{ padding: '4px' }}><Redo2 size={16} /></button>
                  </div>
                  <div style={{ width: '1px', height: '16px', background: 'var(--border-color)', flexShrink: 0 }} />
                  <button style={{ background: 'transparent', border: 'none', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '13px', fontWeight: 500, cursor: 'pointer', flexShrink: 0 }}>
                    Normal <ChevronDown size={14} />
                  </button>
                  <div style={{ width: '1px', height: '16px', background: 'var(--border-color)', flexShrink: 0 }} />
                  <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                    <button className="icon-btn" style={{ padding: '4px' }}><Bold size={16} /></button>
                    <button className="icon-btn" style={{ padding: '4px' }}><Italic size={16} /></button>
                  </div>
                  <div style={{ width: '1px', height: '16px', background: 'var(--border-color)', flexShrink: 0 }} />
                  <div style={{ display: 'flex', gap: '8px', flexShrink: 0 }}>
                    <button className="icon-btn" style={{ padding: '4px' }}><Link size={16} /></button>
                    <button className="icon-btn" style={{ padding: '4px' }}><Code size={16} /></button>
                    <button className="icon-btn" style={{ padding: '4px' }}><Image size={16} /></button>
                  </div>
                  <div style={{ width: '1px', height: '16px', background: 'var(--border-color)', flexShrink: 0 }} />
                  <button className="icon-btn" style={{ padding: '4px', flexShrink: 0 }}><MoreHorizontal size={16} /></button>
                </div>

                <div style={{ flex: 1, padding: '0 16px', overflowY: 'auto' }}>
                  <textarea
                    value={noteContent}
                    onChange={e => setNoteContent(e.target.value)}
                    style={{ width: '100%', height: '100%', background: 'transparent', border: 'none', color: 'var(--text-primary)', fontSize: '15px', resize: 'none', outline: 'none', lineHeight: 1.6 }}
                  />
                </div>

                <div style={{ padding: '16px', borderTop: '1px solid var(--border-color)', marginTop: 'auto' }}>
                  <button style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', background: 'transparent', border: '1px solid var(--border-color)', borderRadius: '24px', color: 'var(--text-primary)', fontSize: '13px', fontWeight: 500, cursor: 'pointer' }} onClick={() => alert('Feature coming soon: Converting note to RAG source document.')}>
                    <FilePlus2 size={16} /> Convert to source
                  </button>
                </div>
              </>
            ) : (
              // STUDIO VIEW
              <>
                <div className="panel-header">
                  <div className="panel-title">Studio</div>
                  <button className="icon-btn" onClick={() => setIsRightSidebarOpen(false)} title="Close Studio"><PanelRight size={18} /></button>
                </div>

                <div className="studio-content">
                  {/* <div className="language-banner">
                    Create an Audio Overview in: हिन्दी, বাংলা, ગુજરાતી, ಕನ್ನಡ, മലയാളം, മറാഠി, ਪੰਜਾਬੀ, தமிழ், తెలుగు
                  </div> */}

                  <div className="studio-grid">
                    <div className="studio-card card-pyq" onClick={() => handleStudioAction('pyq')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><TrendingUp size={18} className="studio-card-icon" style={{ color: '#ffcc99' }} /><span className="studio-card-title">PYQ Analysis</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-audionotes" onClick={() => handleStudioAction('audionotes')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><Headphones size={18} className="studio-card-icon" style={{ color: '#c7a8fa' }} /><span className="studio-card-title">Audio to Notes</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-audio" onClick={() => setIsAudioModalOpen(true)} style={{ cursor: 'pointer' }}>
                      <div className="studio-card-left">
                        <FileAudio size={18} className="studio-card-icon" />
                        <span className="studio-card-title">Audio Overview</span>
                      </div>
                      <ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                    </div>
                    {/* <div className="studio-card card-slide" onClick={() => handleStudioAction('slide')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><PlaySquare size={18} className="studio-card-icon" style={{ color: '#d4e6ba' }} /><span className="studio-card-title">Slide Deck</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div> */}
                    <div className="studio-card card-video" onClick={() => setIsVideoModalOpen(true)} style={{ cursor: 'pointer' }}><div className="studio-card-left"><FileVideo size={18} className="studio-card-icon" style={{ color: '#b6e2c3' }} /><span className="studio-card-title">Video Overview</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-mindmap" onClick={() => handleStudioAction('mindmap')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><Network size={18} className="studio-card-icon" style={{ color: '#e5b3d6' }} /><span className="studio-card-title">Mind Map</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-reports" onClick={() => handleStudioAction('reports')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><FileText size={18} className="studio-card-icon" style={{ color: '#dfc98a' }} /><span className="studio-card-title">Reports</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-flashcards" onClick={() => handleStudioAction('flashcards')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><BrainCircuit size={18} className="studio-card-icon" style={{ color: '#e5b0a3' }} /><span className="studio-card-title">Flashcards</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-quiz" onClick={() => handleStudioAction('quiz')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><MessageSquare size={18} className="studio-card-icon" style={{ color: '#a3d8d3' }} /><span className="studio-card-title">Quiz</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    <div className="studio-card card-infographic" onClick={() => setIsInfographicModalOpen(true)} style={{ cursor: 'pointer' }}><div className="studio-card-left"><FileBarChart size={18} className="studio-card-icon" style={{ color: '#d6b3e5' }} /><span className="studio-card-title">Infographic</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div>
                    {/* <div className="studio-card card-mindmap" onClick={() => setIsMindMapModalOpen(true)} style={{ cursor: 'pointer' }}><div className="studio-card-left"><Network size={18} className="studio-card-icon" style={{ color: '#e5c9b3' }} /><span className="studio-card-title">Mind Map</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div> */}
                    {/* <div className="studio-card card-table" onClick={() => handleStudioAction('table')} style={{ cursor: 'pointer' }}><div className="studio-card-left"><Table size={18} className="studio-card-icon" style={{ color: '#b3bfe5' }} /><span className="studio-card-title">Data Table</span></div><ChevronDown size={16} className="studio-card-arrow" style={{ transform: 'rotate(-90deg)' }} /></div> */}
                  </div>

                  {sessions.length > 0 && (
                    <div style={{ marginTop: '24px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '12px', paddingLeft: '4px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Chat History</div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {sessions.map((s) => (
                          <div
                            key={s.id}
                            onClick={() => { handleSwitchSession(s.id); }}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '12px',
                              padding: '12px 16px',
                              backgroundColor: currentSessionId === s.id ? '#363636' : '#2b2b2b',
                              border: currentSessionId === s.id ? '1px solid var(--accent-color)' : '1px solid rgba(255,255,255,0.05)',
                              borderRadius: '12px',
                              cursor: 'pointer',
                              transition: 'all 0.2s'
                            }}
                          >
                            <MessageSquare size={16} style={{ color: currentSessionId === s.id ? 'var(--accent-color)' : 'var(--text-secondary)' }} />
                            <div style={{ flex: 1, overflow: 'hidden' }}>
                              <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', fontSize: '13px', fontWeight: 500, color: currentSessionId === s.id ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                                {s.title}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {isGeneratingAudio && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--accent-color)' }}>
                      <Loader2 size={20} className="animate-spin" />
                      <span style={{ fontSize: '14px', fontWeight: 500 }}>Generating Podcast...</span>
                    </div>
                  )}
                  {audioOverviewUrl && !isGeneratingAudio && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Your Audio Overview is Ready!</span>
                      <audio controls src={audioOverviewUrl} style={{ width: '100%', height: '40px', outline: 'none' }} />
                    </div>
                  )}

                  {isGeneratingInfographic && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--accent-color)' }}>
                      <Loader2 size={20} className="animate-spin" />
                      <span style={{ fontSize: '14px', fontWeight: 500 }}>Generating Infographic...</span>
                    </div>
                  )}
                  {infographicMarkdown && !isGeneratingInfographic && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Your Interactive Infographic is Ready!</span>
                      </div>
                      <div style={{ maxHeight: '400px', overflowY: 'auto', background: 'var(--bg-panel)', padding: '12px', borderRadius: '8px' }} className="markdown-body text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {infographicMarkdown}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {isGeneratingMindMap && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--accent-color)' }}>
                      <Loader2 size={20} className="animate-spin" />
                      <span style={{ fontSize: '14px', fontWeight: 500 }}>Generating Mind Map...</span>
                    </div>
                  )}
                  {mindMapData && !isGeneratingMindMap && (
                    <div style={{ padding: '8px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', height: '400px', display: 'flex', flexDirection: 'column' }}>
                      <MindMapRenderer data={mindMapData} onNodeClick={handleNodeClick} />
                    </div>
                  )}

                  {isGeneratingVideo && (
                    <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--accent-color)' }}>
                      <Loader2 size={20} className="animate-spin" />
                      <span style={{ fontSize: '14px', fontWeight: 500 }}>Generating Video Overview... (this takes a few minutes)</span>
                    </div>
                  )}
                  {videoOverviewUrl && !isGeneratingVideo && (
                    <div style={{ padding: '8px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px' }}>
                      <video controls style={{ width: '100%', borderRadius: '8px' }} src={videoOverviewUrl} autoPlay />
                    </div>
                  )}

                  {!isGeneratingAudio && !audioOverviewUrl && !isGeneratingInfographic && !infographicMarkdown && !isGeneratingMindMap && !mindMapData && !isGeneratingVideo && !videoOverviewUrl && (
                    <div className="studio-empty">
                      <PenTool size={24} />
                      <div>
                        <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: 4 }}>Studio output will be saved here.</div>
                        <div style={{ fontSize: 13 }}>After adding sources, you can create an Audio Overview, Study Guide, Briefing Doc, and more!</div>
                      </div>
                    </div>
                  )}
                </div>

                <button className="btn-add-note" onClick={() => setIsNoteMode(true)}>
                  <Plus size={18} /> Add note
                </button>
              </>
            )}
          </aside>
        )}

        {/* Center Panel: Chat */}
        <section className="panel chat-center">
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="panel-title">Chat</div>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              {messages.length > 0 && (
                <button
                  onClick={() => handleSwitchSession('new')}
                  className="btn-add-source"
                  style={{ padding: '4px 12px', height: '28px', fontSize: '13px' }}
                >
                  <Plus size={14} /> New Chat
                </button>
              )}
              <button className="icon-btn"><MoreVertical size={18} /></button>
            </div>
          </div>

          <div className="chat-content" style={{ justifyContent: messages.length > 0 ? 'flex-start' : 'center', overflowY: 'auto', paddingBottom: '100px' }}>
            {messages.length === 0 ? (
              <>
                <div className="welcome-icon">👋</div>
                <h1 className="welcome-title">Let's start studying...</h1>
                <p className="welcome-desc">This is your blank canvas to understand, create, or make progress on something new. I can help you get started or you can go ahead and add your own sources.</p>
                <h2 className="suggestions-title">What would you like this notebook to help you do?</h2>
                <div className="suggestions-list">
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the following audio transcript into structured educational notes using Markdown. You are an expert educational assistant that creates clear, structured study notes. You MUST include all five sections with these exact headings: # Title, ## Key Points, ## Explanation, ## Examples, ## Summary in detailed.")}>Audio to notes</button>
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the content from the provided YouTube link/video into structured educational notes using Markdown. You are an expert educational assistant that creates clear, structured study notes. You MUST include all five sections with these exact headings: # Title, ## Key Points, ## Explanation, ## Examples, ## Summary, in detailed format.")}>YouTube to notes</button>
                  <button className="suggestion-btn" onClick={() => handleSendMessage("Learn or understand something")}>Learn or understand something</button>
                </div>
              </>
            ) : (
              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                {messages.map((msg, i) => (
                  <div key={i} style={{ alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%', backgroundColor: msg.role === 'user' ? 'var(--bg-button)' : 'transparent', padding: msg.role === 'user' ? '12px 16px' : '0', borderRadius: '16px', lineHeight: '1.6' }}>
                    {msg.role === 'assistant' && <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--accent-color)' }}><Sparkles size={16} /> <span>StudySnap AI</span></div>}
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
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-color)' }}>
                    <Sparkles size={16} /> <Loader2 size={16} className="animate-spin" /> Thinking...
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="chat-input-container">
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <button
                className="dropdown-btn"
                onClick={() => setIsResponseLangDropdownOpen(!isResponseLangDropdownOpen)}
                style={{ backgroundColor: 'transparent', padding: '6px 12px', borderRadius: '16px', color: 'var(--text-secondary)', border: 'none', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', transition: 'background-color 0.2s', outline: 'none' }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-button-hover)'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                aria-label="Select response language"
              >
                <Globe size={16} />
                <span className="lang-text-full">{responseLanguage}</span>
                <span className="lang-text-short">{responseLanguage.substring(0, 2).toUpperCase()}</span>
                <ChevronDown size={14} />
              </button>
              {isResponseLangDropdownOpen && (
                <div style={{ position: 'absolute', bottom: '100%', left: 0, marginBottom: '12px', background: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '12px', padding: '8px', zIndex: 100, width: '160px', boxShadow: '0 4px 20px rgba(0,0,0,0.5)', maxHeight: '300px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  {["English", "Hinglish", "Manglish", "Hindi", "Marathi"].map(lang => (
                    <div
                      key={lang}
                      onClick={() => { setResponseLanguage(lang); setIsResponseLangDropdownOpen(false); }}
                      style={{ padding: '8px 12px', fontSize: '14px', cursor: 'pointer', borderRadius: '8px', background: responseLanguage === lang ? 'var(--bg-button)' : 'transparent', color: responseLanguage === lang ? 'var(--text-primary)' : 'var(--text-secondary)', transition: 'all 0.2s' }}
                      onMouseOver={(e) => { if (responseLanguage !== lang) e.currentTarget.style.background = 'var(--bg-button-hover)' }}
                      onMouseOut={(e) => e.currentTarget.style.background = responseLanguage === lang ? 'var(--bg-button)' : 'transparent'}
                    >
                      {lang}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--border-color)', margin: '0 4px' }}></div>

            <input
              type="text"
              className="chat-input"
              placeholder="Ask a question or create something"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <span className="input-sources-count">{sources.length} sources</span>
            <button className="btn-send" onClick={() => handleSendMessage()} disabled={isChatting || !inputValue.trim()}>
              <ArrowRight size={16} />
            </button>
          </div>
          <div className="disclaimer">StudySnap AI can be inaccurate; please double check its responses.</div>
        </section>

        {/* Left Sidebar Mini Rail (when closed) */}
        {!isLeftSidebarOpen && (
          <aside className="panel sidebar-left-mini" style={{ width: '64px', minWidth: '64px', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', borderLeft: '1px solid var(--border-color)', backgroundColor: 'var(--bg-panel)' }}>
            <button className="icon-btn" onClick={() => setIsLeftSidebarOpen(true)} title="Open Sources" style={{ marginBottom: '16px' }}>
              <PanelLeft size={18} />
            </button>
            <div style={{ width: '32px', height: '1px', backgroundColor: 'var(--border-color)', marginBottom: '16px', flexShrink: 0 }} />

            <button
              className="icon-btn"
              onClick={() => { setIsLeftSidebarOpen(true); setIsModalOpen(true); }}
              title="Add Sources"
              style={{ color: 'var(--text-primary)' }}
            >
              <Plus size={20} />
            </button>
          </aside>
        )}

        {/* Left Sidebar: Sources */}
        {isLeftSidebarOpen && (
          <aside className="panel sidebar-left">
            <div className="panel-header">
              <div className="panel-title">
                Sources
              </div>
              <button className="icon-btn" onClick={() => setIsLeftSidebarOpen(false)} title="Close Sources">
                <PanelLeft size={18} />
              </button>
            </div>

            <div className="sources-content">
              <button className="btn-add-source" onClick={() => setIsModalOpen(true)}>
                <Plus size={16} />
                Add sources
              </button>

              <div className="search-section" style={{ backgroundColor: '#1d1e20', borderRadius: '16px', padding: '16px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <span className="search-title" style={{ fontSize: '14px', color: 'var(--text-primary)', padding: 0 }}>Search the web for new sources</span>
                <div className="search-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', position: 'relative' }}>

                  {/* Location Dropdown */}
                  <div style={{ position: 'relative' }}>
                    <button
                      className="dropdown-btn"
                      onClick={() => { setIsLocationDropdownOpen(!isLocationDropdownOpen); setIsResearchDropdownOpen(false); }}
                      style={{ backgroundColor: 'var(--bg-panel)', padding: '8px 14px', borderRadius: '24px', color: 'var(--text-primary)' }}
                    >
                      <Globe size={16} /> {searchLocation} <ChevronDown size={14} />
                    </button>
                    {isLocationDropdownOpen && (
                      <div style={{ position: 'absolute', top: '100%', left: 0, marginTop: '4px', background: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '4px', zIndex: 50, width: '120px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
                        <div onClick={() => { setSearchLocation('Web'); setIsLocationDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: searchLocation === 'Web' ? 'var(--bg-button)' : 'transparent', color: 'var(--text-primary)' }}>Web</div>
                        <div onClick={() => { setSearchLocation('Drive'); setIsLocationDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: searchLocation === 'Drive' ? 'var(--bg-button)' : 'transparent', color: 'var(--text-primary)' }}>Drive</div>
                      </div>
                    )}
                  </div>

                  {/* Research Mode Dropdown */}
                  <div style={{ position: 'relative' }}>
                    <button
                      className="dropdown-btn"
                      onClick={() => { setIsResearchDropdownOpen(!isResearchDropdownOpen); setIsLocationDropdownOpen(false); }}
                      style={{ backgroundColor: 'var(--bg-panel)', padding: '8px 14px', borderRadius: '24px', color: 'var(--text-primary)' }}
                    >
                      <div style={{ position: 'relative', width: '16px', height: '16px' }}>
                        <Search size={14} style={{ position: 'absolute', bottom: 0, left: 0 }} />
                        <Sparkles size={8} style={{ position: 'absolute', top: -2, right: -2 }} />
                      </div>
                      {researchMode} <ChevronDown size={14} />
                    </button>
                    {isResearchDropdownOpen && (
                      <div style={{ position: 'absolute', top: '100%', left: 0, marginTop: '4px', background: 'var(--bg-panel)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '4px', zIndex: 50, width: '160px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
                        <div onClick={() => { setResearchMode('Fast Research'); setIsResearchDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: researchMode === 'Fast Research' ? 'var(--bg-button)' : 'transparent', color: 'var(--text-primary)' }}>Fast Research</div>
                        <div onClick={() => { setResearchMode('Deep Research'); setIsResearchDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: researchMode === 'Deep Research' ? 'var(--bg-button)' : 'transparent', color: 'var(--text-primary)' }}>Deep Research</div>
                      </div>
                    )}
                  </div>

                  {!isSearchExpanded ? (
                    <button className="search-input-wrapper" onClick={() => setIsSearchExpanded(true)} style={{ cursor: 'pointer', backgroundColor: 'var(--bg-panel)', borderRadius: '50%', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-color)' }}>
                      <Search size={16} />
                    </button>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', width: '100%', gap: '8px', marginTop: '4px' }}>
                      <input
                        type="text"
                        placeholder={`Enter ${searchLocation} query...`}
                        value={searchBarQuery}
                        onChange={e => setSearchBarQuery(e.target.value)}
                        onKeyDown={e => {
                          if (e.key === 'Enter' && searchBarQuery.trim()) {
                            handleSendMessage(`Perform a ${researchMode} on ${searchLocation} for: "${searchBarQuery}". Return a list of relevant links or a comprehensive briefing document.`);
                            setSearchBarQuery('');
                            setIsSearchExpanded(false);
                          }
                        }}
                        style={{ flex: 1, padding: '10px 14px', borderRadius: '24px', border: '1px solid var(--border-color)', background: 'var(--bg-panel)', color: 'var(--text-primary)', fontSize: '14px', outline: 'none' }}
                        autoFocus
                      />
                      <button className="icon-btn" onClick={() => setIsSearchExpanded(false)}><X size={16} /></button>
                    </div>
                  )}
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
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '16px' }}>
                  {sources.map(src => (
                    <div key={src.id} style={{ padding: '12px', backgroundColor: 'var(--bg-button)', borderRadius: '8px', fontSize: '14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', border: selectedSourceIds.has(src.id) ? '1px solid var(--accent-color)' : '1px solid transparent', opacity: selectedSourceIds.has(src.id) ? 1 : 0.6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                        <input
                          type="checkbox"
                          checked={selectedSourceIds.has(src.id)}
                          onChange={(e) => {
                            const newSet = new Set(selectedSourceIds);
                            if (e.target.checked) newSet.add(src.id);
                            else newSet.delete(src.id);
                            setSelectedSourceIds(newSet);
                          }}
                          style={{ accentColor: 'var(--accent-color)', width: '16px', height: '16px', cursor: 'pointer', flexShrink: 0 }}
                        />
                        <FileText size={16} color="var(--accent-color)" style={{ flexShrink: 0 }} />
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={src.filename}>{src.filename}</span>
                      </div>
                      <button onClick={() => handleRemoveSource(src.id)} className="icon-btn" style={{ width: '24px', height: '24px', flexShrink: 0 }} title="Remove Source">
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        )}
          </>
        )}
      </main>

      <AddSourceModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
        backendUrl={backendUrl}
      />

      <AudioOverviewModal
        isOpen={isAudioModalOpen}
        onClose={() => setIsAudioModalOpen(false)}
        onGenerate={() => {
          handleGenerateAudioOverview();
        }}
      />

      <InfographicModal
        isOpen={isInfographicModalOpen}
        onClose={() => setIsInfographicModalOpen(false)}
        onGenerate={handleGenerateInfographic}
      />

      <MindMapModal
        isOpen={isMindMapModalOpen}
        onClose={() => setIsMindMapModalOpen(false)}
        onGenerate={handleGenerateMindMap}
      />

      <VideoOverviewModal
        isOpen={isVideoModalOpen}
        onClose={() => setIsVideoModalOpen(false)}
        onGenerate={handleGenerateVideoOverview}
      />

      <NoteReaderModal
        note={selectedNoteForModal}
        onClose={() => setSelectedNoteForModal(null)}
        onOpenStudio={() => {
          setSelectedNoteForModal(null);
          setIsLoggedIn(true);
          setActiveTab('studio');
        }}
      />

      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onLoginSuccess={(user) => {
          setCurrentUser(user);
          setIsLoggedIn(true);
          setIsAuthModalOpen(false);
          setActiveTab('home');
        }}
      />

      <UserProfileModal
        isOpen={isProfileModalOpen}
        user={currentUser}
        onClose={() => setIsProfileModalOpen(false)}
        onLogout={() => setIsLoggedIn(false)}
      />

      <BottomNav
        activeTab={activeTab}
        isLoggedIn={isLoggedIn}
        onTabChange={(tab) => {
          setActiveTab(tab);
        }}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        onOpenAccount={() => setIsProfileModalOpen(true)}
      />
    </div>
  );
}

export default App;
