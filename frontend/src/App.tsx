import { useState, useEffect, useRef, useCallback } from 'react';
import AddSourceModal from './AddSourceModal';
import AudioOverviewModal from './AudioOverviewModal';

import InfographicModal from './InfographicModal';
import MindMapModal from './MindMapModal';
import MindMapRenderer from './MindMapRenderer';
import { InfographicRenderer } from './components/InfographicEditor/InfographicRenderer';
import LandingPage from './components/LandingPage';
import StudentDashboard from './components/StudentDashboard';
import NoteReaderModal from './components/NoteReaderModal';
import AuthModal from './components/AuthModal';
import UserProfileModal from './components/UserProfileModal';
import BottomNav from './components/BottomNav';
import InteractiveParticleBackground from './components/InteractiveParticleBackground';
import { QuizPage } from './components/QuizPage';
import { AudioOverviewPage } from './components/AudioOverviewPage';
import { MindMapPage } from './components/MindMapPage';
import { FlashcardsPage } from './components/FlashcardsPage';
import { ReportsPage } from './components/ReportsPage';
import type { PublicNote } from './data/mockNotes';
import { fetchStudentProfile } from './config/supabase';
import {
  Search, Plus, X, ArrowLeft,
  ChevronDown, FileText, FileAudio, Network,
  BrainCircuit, MessageSquare, Sparkles,
  ArrowRight, MoreVertical, PanelLeft, PanelRight,
  PenTool, Loader2, Globe, TrendingUp, Headphones,
  ChevronRight, Maximize2, Trash2, Undo2, Redo2, Bold, Italic, Link, Code, Image, MoreHorizontal, FilePlus2,
  Check, Cloud, CheckCircle, AlertCircle
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { isSupabaseConfigured, supabase, getCurrentSession, signOutUser } from './config/supabase';
import { aiFetch } from './config/api';
import { useTheme } from './context/ThemeContext';
import './index.css';

/**
 * Normalizes malformed markdown from local LLMs/SLMs:
 * - Fixes enclosing title tags: '#Title## Understanding...' -> '# Understanding...'
 * - Fixes enclosing header tags: '##KeyPoints##' -> '## Key Points'
 * - Fixes missing space after hashes: '##KeyPoints' -> '## Key Points'
 * - Converts bullet characters (•, ●) to markdown lists ('- ')
 */
export function preprocessMarkdown(text: string): string {
  if (!text) return "";

  return text
    // 1. Fix enclosing title tags with title text on the same line:
    .replace(/^#{1,6}\s*Title\s*#{1,6}\s*:?\s*(.+)$/gim, '# $1')
    .replace(/^#{1,6}\s*Title\s*:\s*(.+)$/gim, '# $1')

    // 2. Fix enclosing tags alone on a line:
    .replace(/^(#{1,6})\s*([A-Za-z0-9_ ]+?)\s*#{1,6}\s*$/gm, (_, hashes, heading) => {
      const formatted = heading.replace(/([a-z])([A-Z])/g, '$1 $2').trim();
      return `${hashes} ${formatted}`;
    })

    // 3. Fix general enclosing tags with non-empty text after trailing hashes:
    .replace(/^(#{1,6})\s*([A-Za-z0-9_ ]+?)\s*#{1,6}\s*:?\s*([^#\s].*)$/gm, (_, hashes, heading, rest) => {
      const formatted = heading.replace(/([a-z])([A-Z])/g, '$1 $2').trim();
      return `${hashes} ${formatted}: ${rest}`;
    })

    // 4. Fix missing space after hashes at start of line:
    .replace(/^(#{1,6})([A-Za-z0-9])/gm, '$1 $2')

    // 5. Split CamelCase in standalone headings:
    .replace(/^(#{1,6}\s+)([A-Za-z]+)$/gm, (_, prefix, word) => {
      const splitWord = word.replace(/([a-z])([A-Z])/g, '$1 $2');
      return `${prefix}${splitWord}`;
    })

    // 6. Convert bullet characters like • or ● to markdown list syntax "- "
    .replace(/^[•●]\s+/gm, '- ');
}

function App() {
  const { setUserContext, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState<'home' | '' | 'chat' | 'infographic' | 'quiz' | 'audio' | 'mindmap' | 'reports' | 'flashcards'>('home');
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
  const [isChatting, setIsChatting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const adjustTextareaHeight = useCallback(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    // Reset height to 'auto' first so scrollHeight accurately shrinks when text is deleted
    textarea.style.height = 'auto';

    const minHeight = 24;
    const maxHeight = 200;
    const scrollHeight = textarea.scrollHeight;

    if (scrollHeight > maxHeight) {
      textarea.style.height = `${maxHeight}px`;
      textarea.style.overflowY = 'auto';
    } else {
      textarea.style.height = `${Math.max(scrollHeight, minHeight)}px`;
      textarea.style.overflowY = 'hidden';
    }
  }, []);

  useEffect(() => {
    adjustTextareaHeight();
  }, [inputValue, adjustTextareaHeight]);

  useEffect(() => {
    if (messages.length > 0) {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isChatting]);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAudioModalOpen, setIsAudioModalOpen] = useState(false);
  const [isNoteMode, setIsNoteMode] = useState(false);
  const [noteTitle, setNoteTitle] = useState('New Note');
  const [noteContent, setNoteContent] = useState('');
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  const [audioOverviewUrl, setAudioOverviewUrl] = useState<string | null>(null);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [infographicMarkdown, setInfographicMarkdown] = useState<string | null>(null);
  const [infographicData, setInfographicData] = useState<any>(null);
  const [isGeneratingInfographic, setIsGeneratingInfographic] = useState(false);
  const [isInfographicModalOpen, setIsInfographicModalOpen] = useState(false);
  const [isMindMapModalOpen, setIsMindMapModalOpen] = useState(false);
  const [mindMapData, setMindMapData] = useState<any>(null);
  const [isGeneratingMindMap, setIsGeneratingMindMap] = useState(false);

  // 2-State System & Navigation States
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState<any | null>({
    id: 'demo-student',
    name: 'Rohan Sharma',
    full_name: 'Rohan Sharma',
    email: 'rohan.sharma@iitb.ac.in',
    college: 'DKTE Society\'s Textile and Engineering Institute',
    college_name: 'DKTE Society\'s Textile and Engineering Institute',
    degree_program: 'B.Tech / B.E.',
    branch: 'Computer Science Engineering (CSE)',
    academic_year: 'Third Year',
    semester: 'Semester 5',
    graduation_year: '2026',
    roll_number: '22CS1049',
    email_verified: true,
  });
  const [selectedNoteForModal, setSelectedNoteForModal] = useState<PublicNote | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);

  // Active Notebook & Autosave State
  const [activeNotebook, setActiveNotebook] = useState<any | null>(null);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const backendUrl = "http://localhost:8000";

  // Supabase Auth Session Synchronization on Mount
  useEffect(() => {
    fetchSources();

    if (isSupabaseConfigured() && supabase) {
      getCurrentSession().then(async session => {
        if (session?.user) {
          setIsLoggedIn(true);
          const p = await fetchStudentProfile(session.user.id).catch(() => null);
          if (p) {
            setCurrentUser(p);
          } else {
            setCurrentUser({
              id: session.user.id,
              name: session.user.user_metadata?.full_name || session.user.user_metadata?.name || session.user.email?.split('@')[0] || 'Student',
              full_name: session.user.user_metadata?.full_name || session.user.user_metadata?.name || 'Student',
              email: session.user.email || '',
              college: session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
              college_name: session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
              degree_program: session.user.user_metadata?.degree_program || 'B.Tech / B.E.',
              branch: session.user.user_metadata?.branch || 'Computer Science Engineering (CSE)',
              academic_year: session.user.user_metadata?.academic_year || 'Third Year',
              semester: session.user.user_metadata?.semester || 'Semester 5',
              graduation_year: session.user.user_metadata?.graduation_year || '2026',
              email_verified: Boolean(session.user.email_confirmed_at),
            });
          }
        }
      });

      const { data: { subscription } } = supabase.auth.onAuthStateChange(async (_event, session) => {
        if (session?.user) {
          setIsLoggedIn(true);
          const p = await fetchStudentProfile(session.user.id).catch(() => null);
          if (p) {
            setCurrentUser(p);
          } else {
            setCurrentUser({
              id: session.user.id,
              name: session.user.user_metadata?.full_name || session.user.user_metadata?.name || session.user.email?.split('@')[0] || 'Student',
              full_name: session.user.user_metadata?.full_name || session.user.user_metadata?.name || 'Student',
              email: session.user.email || '',
              college: session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
              college_name: session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
              degree_program: session.user.user_metadata?.degree_program || 'B.Tech / B.E.',
              branch: session.user.user_metadata?.branch || 'Computer Science Engineering (CSE)',
              academic_year: session.user.user_metadata?.academic_year || 'Third Year',
              semester: session.user.user_metadata?.semester || 'Semester 5',
              graduation_year: session.user.user_metadata?.graduation_year || '2026',
              email_verified: Boolean(session.user.email_confirmed_at),
            });
          }
        } else if (_event === 'SIGNED_OUT') {
          setIsLoggedIn(false);
          setCurrentUser(null);
          setActiveNotebook(null);
        }
      });

      return () => {
        subscription.unsubscribe();
      };
    } else {
      const stored = localStorage.getItem('studysnap_user');
      if (stored) {
        try {
          const u = JSON.parse(stored);
          if (u && u.email) {
            setCurrentUser(u);
            setIsLoggedIn(true);
          }
        } catch { }
      }
    }
  }, []);

  // Synchronize authenticated user ID and remote theme preference with ThemeContext
  useEffect(() => {
    if (currentUser?.id) {
      setUserContext(currentUser.id);
      if (currentUser.theme_preference) {
        setTheme(currentUser.theme_preference, false);
      }
    } else {
      setUserContext(null);
    }
  }, [currentUser?.id, currentUser?.theme_preference, setUserContext, setTheme]);

  const handleOpenNotebook = (nb: any) => {
    setActiveNotebook(nb);
    setNoteTitle(nb.title || 'Untitled Notebook');
    setNoteContent(nb.content || '');
    if (Array.isArray(nb.sources) && nb.sources.length > 0) {
      setSources(nb.sources);
      setSelectedSourceIds(new Set(nb.sources.map((s: any) => s.id)));
    }
    if (Array.isArray(nb.chat_history) && nb.chat_history.length > 0) {
      setMessages(nb.chat_history);
    } else {
      setMessages([]);
    }
    setActiveTab('');
  };

  // Debounced Autosave for Active Notebook in MongoDB
  useEffect(() => {
    if (!activeNotebook?.id) return;

    setSaveStatus('saving');
    const timer = setTimeout(async () => {
      try {
        const res = await aiFetch(`/api/notebooks/${activeNotebook.id}`, {
          method: 'PUT',
          body: JSON.stringify({
            title: noteTitle,
            content: noteContent,
            sources: sources,
            chat_history: messages,
            sources_count: sources.length,
            notes_count: noteContent.trim() ? 1 : 0
          })
        });

        if (res.ok) {
          setSaveStatus('saved');
        } else {
          setSaveStatus('error');
        }
      } catch (err) {
        console.warn('[Autosave] Failed to autosave notebook:', err);
        setSaveStatus('error');
      }
    }, 1200);

    return () => clearTimeout(timer);
  }, [noteTitle, noteContent, sources, messages, activeNotebook?.id]);

  const handleLogout = async () => {
    await signOutUser();
    setIsLoggedIn(false);
    setCurrentUser(null);
    setActiveNotebook(null);
    setActiveTab('home');
  };

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
    if (isChatting) return;
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
          response_language: "English"
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
      const errorMessages = [...newMessages, { role: "assistant", content: "Sorry, I encountered an error. Please ensure LM  is running on port 1234 and the backend is running." }];
      setMessages(errorMessages);
      setSessions(prev => prev.map(s => s.id === sessionIdToUse ? { ...s, messages: errorMessages } : s));
    } finally {
      setIsChatting(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (e.nativeEvent.isComposing) return;
      e.preventDefault();
      if (!isChatting && inputValue.trim()) {
        handleSendMessage();
      }
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
          response_language: "English"
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



  const handleGenerateInfographic = async (style: string, detailLevel: string, customPrompt: string) => {
    setIsGeneratingInfographic(true);
    setInfographicMarkdown(null);
    setInfographicData(null);
    try {
      const res = await fetch(`${backendUrl}/api/infographic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_source_ids: Array.from(selectedSourceIds),
          style,
          detail_level: detailLevel,
          custom_prompt: customPrompt,
          response_language: "English"
        }),
      });
      const data = await res.json();
      if (res.ok && data.success && data.data) {
        setInfographicData(data.data);
      } else if (res.ok && data.success && data.markdown) {
        setInfographicMarkdown(data.markdown);
      } else {
        alert("Failed to generate infographic from backend: " + (data.detail || "Unknown error"));
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
          response_language: "English"
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

  const handleAction = (type: string) => {
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
      case 'audionotes': prompt = "Convert the provided audio transcript into structured educational notes using clean Markdown. You are an expert educational assistant that creates clear, structured study notes. Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format."; break;
    }
    if (prompt) {
      handleSendMessage(prompt);
    }
  };

  return (
    <div className="app-container">
      {/* Full-Page Interactive Cursor-Reactive Particle Background */}
      <InteractiveParticleBackground />

      {/* Main Content Area */}
      <main className="main-content" style={{ position: 'relative', zIndex: 1, overflowY: activeTab === 'home' || activeTab === 'infographic' ? 'auto' : 'hidden' }}>
        { activeTab === 'quiz' ? (
          <QuizPage backendUrl={backendUrl} selectedSourceIds={selectedSourceIds} responseLanguage="English" onBack={() => setActiveTab('')} />
        ) : activeTab === 'audio' ? (
          <AudioOverviewPage backendUrl={backendUrl} selectedSourceIds={selectedSourceIds} responseLanguage="English" onBack={() => setActiveTab('')} />
        ) : activeTab === 'mindmap' ? (
          <MindMapPage backendUrl={backendUrl} selectedSourceIds={selectedSourceIds} responseLanguage="English" onBack={() => setActiveTab('')} onNodeClick={handleNodeClick} />
        ) : activeTab === 'flashcards' ? (
          <FlashcardsPage backendUrl={backendUrl} selectedSourceIds={selectedSourceIds} responseLanguage="English" onBack={() => setActiveTab('')} />
        ) : activeTab === 'reports' ? (
          <ReportsPage onBack={() => setActiveTab('')} />
        ) : activeTab === 'home' ? (
          !isLoggedIn ? (
            <LandingPage
              onCreateNotebook={() => {
                if (isLoggedIn) {
                  setActiveTab('');
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
              onOpenStudio={() => setActiveTab('')}
              onOpenNotebook={handleOpenNotebook}
              onLogout={handleLogout}
              onOpenProfile={() => setIsProfileModalOpen(true)}
            />
          )
        ) : (
          <>
            {/* Top Bar for Workspace View */}
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '42px',
              backgroundColor: 'var(--surface-primary, var(--bg-panel))',
              borderBottom: '1px solid var(--border-default)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 16px',
              zIndex: 30
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <button
                  onClick={() => setActiveTab('home')}
                  style={{
                    background: 'var(--surface-secondary, rgba(255, 255, 255, 0.08))',
                    border: '1px solid var(--border-default)',
                    color: 'var(--text-primary)',
                    padding: '4px 12px',
                    borderRadius: '9999px',
                    fontSize: '0.8rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    transition: 'all 0.2s'
                  }}
                >
                  <ArrowLeft size={14} />
                  Back to Dashboard
                </button>

                {activeNotebook && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem' }}>
                    <span style={{
                      background: 'var(--selected-bg, rgba(59, 130, 246, 0.15))',
                      color: 'var(--accent-hover, #60A5FA)',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontWeight: 700,
                      fontSize: '0.75rem'
                    }}>
                      {activeNotebook.code || 'CS101'}
                    </span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 600, maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {noteTitle || activeNotebook.title}
                    </span>
                  </div>
                )}
              </div>

              {/* Autosave Status Indicator */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                {activeNotebook && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem' }}>
                    {saveStatus === 'saving' && (
                      <span style={{ color: 'var(--warning, #F59E0B)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Loader2 size={13} className="animate-spin" />
                        Saving...
                      </span>
                    )}
                    {saveStatus === 'saved' && (
                      <span style={{ color: 'var(--success, #10B981)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Check size={13} />
                        Saved to Supabase
                      </span>
                    )}
                    {saveStatus === 'error' && (
                      <span style={{ color: 'var(--danger, #EF4444)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Cloud size={13} />
                        Save pending
                      </span>
                    )}
                  </div>
                )}

                <div style={{ fontSize: '0.825rem', color: 'var(--accent-hover, #F59E0B)', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Sparkles size={15} />
                  AI Study Assistant & RAG
                </div>

                {isLoggedIn && (
                  <button
                    onClick={() => setIsProfileModalOpen(true)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      background: 'var(--surface-secondary, rgba(255, 255, 255, 0.06))',
                      border: '1px solid var(--border-default)',
                      borderRadius: '9999px',
                      padding: '3px 10px 3px 4px',
                      cursor: 'pointer',
                      color: 'var(--text-primary)',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                    }}
                    title="Open Student Profile"
                  >
                    <div style={{
                      width: '24px',
                      height: '24px',
                      borderRadius: '50%',
                      overflow: 'hidden',
                      backgroundColor: 'var(--button-primary-bg, var(--accent-primary))',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'var(--button-primary-text, var(--accent-contrast))',
                      fontSize: '0.75rem',
                      fontWeight: 700
                    }}>
                      {currentUser?.avatar_url ? (
                        <img src={currentUser.avatar_url} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      ) : (
                        (currentUser?.full_name?.charAt(0) || currentUser?.name?.charAt(0) || 'S').toUpperCase()
                      )}
                    </div>
                    <span>{currentUser?.full_name?.split(' ')[0] || currentUser?.name?.split(' ')[0] || 'Profile'}</span>
                  </button>
                )}
              </div>
            </div>

            {/* Right Sidebar Mini Rail (when closed) */}
            {!isRightSidebarOpen && (
              <aside className="panel sidebar-right-mini" style={{ width: '64px', minWidth: '64px', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', borderRight: '1px solid var(--border-default)', backgroundColor: 'var(--surface-primary, var(--bg-panel))' }}>
                <button className="icon-btn" onClick={() => setIsRightSidebarOpen(true)} title="Open Tools" style={{ marginBottom: '16px' }}>
                  <PanelRight size={18} />
                </button>
                <div style={{ width: '32px', height: '1px', backgroundColor: 'var(--border-default)', marginBottom: '16px', flexShrink: 0 }} />

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', overflowX: 'hidden', padding: '0 8px', scrollbarWidth: 'none', flex: 1, alignItems: 'center' }}>
                  <button className="mini-rail-btn" onClick={() => { setIsRightSidebarOpen(true); handleAction('pyq'); }} title="PYQ Analysis">
                    <TrendingUp size={18} style={{ color: 'var(--card-pyq-accent, #F59E0B)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                  <button className="mini-rail-btn" onClick={() => { setActiveTab('audio'); }} title="Audio Overview">
                    <FileAudio size={18} style={{ color: 'var(--card-audio-accent, #3B82F6)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                  <button className="mini-rail-btn" onClick={() => { setIsRightSidebarOpen(true); handleAction('audionotes'); }} title="Audio to Notes">
                    <Headphones size={18} style={{ color: 'var(--card-audionotes-accent, #A855F7)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>

                  <button className="mini-rail-btn" onClick={() => { setActiveTab('reports'); }} title="Reports">
                    <FileText size={18} style={{ color: 'var(--card-reports-accent, #EAB308)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                  <button className="mini-rail-btn" onClick={() => { setActiveTab('flashcards'); }} title="Flashcards">
                    <BrainCircuit size={18} style={{ color: 'var(--card-flashcards-accent, #F43F5E)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                  <button className="mini-rail-btn" onClick={() => { setActiveTab('quiz'); }} title="Quiz">
                    <MessageSquare size={18} style={{ color: 'var(--card-quiz-accent, #10B981)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                  <button className="mini-rail-btn" onClick={() => { setActiveTab('mindmap'); }} title="Mind Map">
                    <Network size={18} style={{ color: 'var(--card-mindmap-accent, #EC4899)' }} />
                    <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'var(--text-muted)' }} />
                  </button>
                </div>

                <button onClick={() => { setIsRightSidebarOpen(true); setIsNoteMode(true); }} style={{ width: '40px', height: '40px', borderRadius: '50%', backgroundColor: 'var(--button-primary-bg, var(--color-cta-bg))', color: 'var(--button-primary-text, var(--color-cta-text))', border: '1px solid var(--border-default)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginTop: '16px', flexShrink: 0, boxShadow: '0 4px 12px rgba(0,0,0,0.2)', cursor: 'pointer', transition: 'all 0.2s' }} title="Add Note">
                  <PenTool size={18} />
                  <Plus size={10} strokeWidth={3} style={{ position: 'absolute', bottom: 4, right: 4, color: 'currentColor' }} />
                </button>
              </aside>
            )}

            {/* Right Sidebar:  / Note */}
            {isRightSidebarOpen && (
              <aside className="panel sidebar-right" style={{ display: 'flex', flexDirection: 'column' }}>

                {isNoteMode ? (
                  // NOTE EDITOR VIEW
                  <>
                    <div className="panel-header" style={{ padding: '12px 16px', borderBottom: 'none' }}>
                      <div style={{ display: 'flex', alignItems: 'center', fontSize: '13px', color: 'var(--text-secondary)', cursor: 'pointer' }} onClick={() => setIsNoteMode(false)}>
                        <ChevronRight size={14} style={{ margin: '0 4px' }} /> <span style={{ color: 'var(--text-primary)' }}>Note</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        {saveStatus === 'saving' && (
                          <span style={{ fontSize: '11px', color: '#f59e0b', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                            <Loader2 size={12} className="animate-spin" /> Saving...
                          </span>
                        )}
                        {saveStatus === 'saved' && (
                          <span style={{ fontSize: '11px', color: '#10b981', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                            <CheckCircle size={12} /> Saved
                          </span>
                        )}
                        {saveStatus === 'error' && (
                          <span style={{ fontSize: '11px', color: '#ef4444', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                            <AlertCircle size={12} /> Save error
                          </span>
                        )}
                        <button className="icon-btn" title="Expand"><Maximize2 size={14} /></button>
                      </div>
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
                  //  VIEW
                  <>
                    <div className="panel-header">
                      <div className="panel-title"></div>
                      <button className="icon-btn" onClick={() => setIsRightSidebarOpen(false)} title="Close "><PanelRight size={18} /></button>
                    </div>

                    <div className="-content">

                      <div className="-grid">
                        <div className="-card card-pyq" onClick={() => handleAction('pyq')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <TrendingUp size={18} className="-card-icon" />
                            <span className="-card-title">PYQ Analysis</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
                        <div className="-card card-audionotes" onClick={() => handleAction('audionotes')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <Headphones size={18} className="-card-icon" />
                            <span className="-card-title">Audio to Notes</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
                        <div className="-card card-audio" onClick={() => setActiveTab('audio')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <FileAudio size={18} className="-card-icon" />
                            <span className="-card-title">Audio Overview</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>

                        <div className="-card card-mindmap" onClick={() => setActiveTab('mindmap')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <Network size={18} className="-card-icon" />
                            <span className="-card-title">Mind Map</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
                        <div className="-card card-reports" onClick={() => setActiveTab('reports')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <FileText size={18} className="-card-icon" />
                            <span className="-card-title">Reports</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
                        <div className="-card card-flashcards" onClick={() => setActiveTab('flashcards')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <BrainCircuit size={18} className="-card-icon" />
                            <span className="-card-title">Flashcards</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
                        <div className="-card card-quiz" onClick={() => setActiveTab('quiz')} style={{ cursor: 'pointer' }}>
                          <div className="-card-left">
                            <MessageSquare size={18} className="-card-icon" />
                            <span className="-card-title">Quiz</span>
                          </div>
                          <ChevronDown size={16} className="-card-arrow" style={{ transform: 'rotate(-90deg)' }} />
                        </div>
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
                                  backgroundColor: currentSessionId === s.id ? 'var(--selected-bg)' : 'var(--surface-secondary, var(--bg-button))',
                                  border: currentSessionId === s.id ? '1px solid var(--accent-primary, var(--accent-color))' : '1px solid var(--border-default)',
                                  borderRadius: '12px',
                                  cursor: 'pointer',
                                  transition: 'all 0.2s'
                                }}
                              >
                                <MessageSquare size={16} style={{ color: currentSessionId === s.id ? 'var(--accent-primary, var(--accent-color))' : 'var(--text-secondary)' }} />
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
                      {infographicData && !isGeneratingInfographic && (
                        <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Your Visual Infographic is Ready!</span>
                          </div>
                          <div style={{ maxHeight: '600px', overflowY: 'auto', background: 'var(--bg-panel)', padding: '12px', borderRadius: '8px' }}>
                            <InfographicRenderer data={infographicData} backendUrl={backendUrl} />
                          </div>
                        </div>
                      )}
                      {infographicMarkdown && !isGeneratingInfographic && !infographicData && (
                        <div style={{ padding: '16px', background: 'var(--bg-button)', borderRadius: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>Your Interactive Infographic is Ready!</span>
                          </div>
                          <div style={{ maxHeight: '400px', overflowY: 'auto', background: 'var(--bg-panel)', padding: '12px', borderRadius: '8px' }} className="markdown-body text-sm">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {preprocessMarkdown(infographicMarkdown)}
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



                      {!isGeneratingAudio && !audioOverviewUrl && !isGeneratingInfographic && !infographicMarkdown && !infographicData && !isGeneratingMindMap && !mindMapData && (
                        <div className="-empty">
                          <PenTool size={24} />
                          <div>
                            <div style={{ fontWeight: 500, color: 'var(--text-primary)', marginBottom: 4 }}> output will be saved here.</div>
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

              <div className="chat-content" style={{ justifyContent: messages.length > 0 ? 'flex-start' : 'center', overflowY: 'auto', paddingBottom: '240px' }}>
                {messages.length === 0 ? (
                  <>
                    <div className="welcome-icon">👋</div>
                    <h1 className="welcome-title">Let's start studying...</h1>
                    <p className="welcome-desc">This is your blank canvas to understand, create, or make progress on something new. I can help you get started or you can go ahead and add your own sources.</p>
                    <h2 className="suggestions-title">What would you like this notebook to help you do?</h2>
                    <div className="suggestions-list">
                      <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the following audio transcript into structured educational notes using clean Markdown. You are an expert educational assistant that creates clear, structured study notes. Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format.")}>Audio to notes</button>
                      <button className="suggestion-btn" onClick={() => handleSendMessage("Convert the content from the provided YouTube link/video into structured educational notes using clean Markdown. You are an expert educational assistant that creates clear, structured study notes. Format all sections with bold Markdown headers: # [Topic Title], ## Key Points, ## Explanation, ## Examples, and ## Summary in detailed format.")}>YouTube to notes</button>
                      <button className="suggestion-btn" onClick={() => handleSendMessage("Learn or understand something")}>Learn or understand something</button>
                    </div>
                  </>
                ) : (
                  <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '24px' }}>
                    {messages.map((msg, i) => (
                      <div key={i} style={{
                        alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '80%',
                        backgroundColor: msg.role === 'user' ? 'var(--chat-user-bg, var(--surface-secondary))' : 'transparent',
                        border: msg.role === 'user' ? '1px solid var(--border-default)' : 'none',
                        color: 'var(--text-primary)',
                        padding: msg.role === 'user' ? '12px 16px' : '0',
                        borderRadius: '16px',
                        lineHeight: '1.6',
                        whiteSpace: msg.role === 'user' ? 'pre-wrap' : 'normal',
                        wordBreak: 'break-word',
                        overflowWrap: 'anywhere'
                      }}>
                        {msg.role === 'assistant' && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', color: 'var(--accent-primary, var(--accent-color))' }}>
                            <Sparkles size={16} /> <span>StudySnap AI</span>
                          </div>
                        )}
                        <div className={msg.role === 'assistant' ? 'markdown-body' : ''}>
                          {msg.role === 'assistant' ? (
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{preprocessMarkdown(msg.content)}</ReactMarkdown>
                          ) : (
                            msg.content
                          )}
                        </div>
                      </div>
                    ))}
                    {isChatting && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-primary, var(--accent-color))' }}>
                        <Sparkles size={16} /> <Loader2 size={16} className="animate-spin" /> Thinking...
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                )}
              </div>

              <div className="chat-input-container">
                <textarea
                  ref={textareaRef}
                  className="chat-input"
                  placeholder="Ask a question or create something"
                  value={inputValue}
                  rows={1}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
                <div className="chat-input-actions">
                  <span className="input-sources-count">{sources.length} sources</span>
                  <button
                    className="btn-send"
                    onClick={() => handleSendMessage()}
                    disabled={isChatting || !inputValue.trim()}
                    aria-label="Send message"
                    title={isChatting ? "Thinking..." : "Send message"}
                  >
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>
              <div className="disclaimer">StudySnap AI can be inaccurate; please double check its responses.</div>
            </section>

            {/* Left Sidebar Mini Rail (when closed) */}
            {!isLeftSidebarOpen && (
              <aside className="panel sidebar-left-mini" style={{ width: '64px', minWidth: '64px', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', borderLeft: '1px solid var(--border-default)', backgroundColor: 'var(--surface-primary, var(--bg-panel))' }}>
                <button className="icon-btn" onClick={() => setIsLeftSidebarOpen(true)} title="Open Sources" style={{ marginBottom: '16px' }}>
                  <PanelLeft size={18} />
                </button>
                <div style={{ width: '32px', height: '1px', backgroundColor: 'var(--border-default)', marginBottom: '16px', flexShrink: 0 }} />

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

                  <div className="search-section" style={{ backgroundColor: 'var(--surface-secondary, var(--bg-panel))', borderRadius: '16px', padding: '16px', border: '1px solid var(--border-default)', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <span className="search-title" style={{ fontSize: '14px', color: 'var(--text-primary)', padding: 0 }}>Search the web for new sources</span>
                    <div className="search-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', position: 'relative' }}>

                      {/* Location Dropdown */}
                      <div style={{ position: 'relative' }}>
                        <button
                          className="dropdown-btn"
                          onClick={() => { setIsLocationDropdownOpen(!isLocationDropdownOpen); setIsResearchDropdownOpen(false); }}
                          style={{ backgroundColor: 'var(--surface-primary, var(--bg-panel))', border: '1px solid var(--border-default)', padding: '8px 14px', borderRadius: '24px', color: 'var(--text-primary)' }}
                        >
                          <Globe size={16} /> {searchLocation} <ChevronDown size={14} />
                        </button>
                        {isLocationDropdownOpen && (
                          <div style={{ position: 'absolute', top: '100%', left: 0, marginTop: '4px', background: 'var(--surface-elevated, var(--bg-panel))', border: '1px solid var(--border-default)', borderRadius: '8px', padding: '4px', zIndex: 50, width: '120px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
                            <div onClick={() => { setSearchLocation('Web'); setIsLocationDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: searchLocation === 'Web' ? 'var(--selected-bg)' : 'transparent', color: 'var(--text-primary)' }}>Web</div>
                            <div onClick={() => { setSearchLocation('Drive'); setIsLocationDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: searchLocation === 'Drive' ? 'var(--selected-bg)' : 'transparent', color: 'var(--text-primary)' }}>Drive</div>
                          </div>
                        )}
                      </div>

                      {/* Research Mode Dropdown */}
                      <div style={{ position: 'relative' }}>
                        <button
                          className="dropdown-btn"
                          onClick={() => { setIsResearchDropdownOpen(!isResearchDropdownOpen); setIsLocationDropdownOpen(false); }}
                          style={{ backgroundColor: 'var(--surface-primary, var(--bg-panel))', border: '1px solid var(--border-default)', padding: '8px 14px', borderRadius: '24px', color: 'var(--text-primary)' }}
                        >
                          <div style={{ position: 'relative', width: '16px', height: '16px' }}>
                            <Search size={14} style={{ position: 'absolute', bottom: 0, left: 0 }} />
                            <Sparkles size={8} style={{ position: 'absolute', top: -2, right: -2 }} />
                          </div>
                          {researchMode} <ChevronDown size={14} />
                        </button>
                        {isResearchDropdownOpen && (
                          <div style={{ position: 'absolute', top: '100%', left: 0, marginTop: '4px', background: 'var(--surface-elevated, var(--bg-panel))', border: '1px solid var(--border-default)', borderRadius: '8px', padding: '4px', zIndex: 50, width: '160px', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }}>
                            <div onClick={() => { setResearchMode('Fast Research'); setIsResearchDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: researchMode === 'Fast Research' ? 'var(--selected-bg)' : 'transparent', color: 'var(--text-primary)' }}>Fast Research</div>
                            <div onClick={() => { setResearchMode('Deep Research'); setIsResearchDropdownOpen(false); }} style={{ padding: '8px 12px', fontSize: '13px', cursor: 'pointer', borderRadius: '4px', background: researchMode === 'Deep Research' ? 'var(--selected-bg)' : 'transparent', color: 'var(--text-primary)' }}>Deep Research</div>
                          </div>
                        )}
                      </div>

                      {!isSearchExpanded ? (
                        <button className="search-input-wrapper" onClick={() => setIsSearchExpanded(true)} style={{ cursor: 'pointer', backgroundColor: 'var(--surface-primary, var(--bg-panel))', borderRadius: '50%', width: '36px', height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid var(--border-default)', color: 'var(--text-primary)' }}>
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
                            style={{ flex: 1, padding: '10px 14px', borderRadius: '24px', border: '1px solid var(--border-default)', background: 'var(--surface-input, var(--bg-panel))', color: 'var(--text-primary)', fontSize: '14px', outline: 'none' }}
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
                        <div key={src.id} style={{
                          padding: '12px',
                          backgroundColor: 'var(--source-card-bg, var(--surface-secondary))',
                          borderRadius: '8px',
                          fontSize: '14px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          border: selectedSourceIds.has(src.id) ? '1px solid var(--accent-primary, var(--accent-color))' : '1px solid var(--border-subtle)',
                          color: 'var(--text-primary)',
                          opacity: selectedSourceIds.has(src.id) ? 1 : 0.75,
                          transition: 'all 0.2s ease'
                        }}>
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
                              style={{ accentColor: 'var(--accent-primary, var(--accent-color))', width: '16px', height: '16px', cursor: 'pointer', flexShrink: 0 }}
                            />
                            <FileText size={16} color="var(--accent-primary, var(--accent-color))" style={{ flexShrink: 0 }} />
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



      <NoteReaderModal
        note={selectedNoteForModal}
        onClose={() => setSelectedNoteForModal(null)}
        onOpenStudio={() => {
          setSelectedNoteForModal(null);
          setIsLoggedIn(true);
          setActiveTab('');
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
        onLogout={handleLogout}
        onProfileUpdated={(updated) => {
          setCurrentUser(updated);
          localStorage.setItem('studysnap_user', JSON.stringify(updated));
        }}
      />

      <BottomNav
        activeTab={activeTab}
        isLoggedIn={isLoggedIn}
        onTabChange={(tab) => {
          setActiveTab(tab);
        }}
        onOpenAuth={() => setIsAuthModalOpen(true)}
        onOpenAccount={() => setIsProfileModalOpen(true)}
        currentUser={currentUser}
      />
    </div>
  );
}

export default App;
