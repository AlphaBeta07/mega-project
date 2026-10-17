const chatbotService = require('../services/chatbotService');
const ChatbotSession = require('../models/ChatbotSession');
const AIArtifact = require('../models/AIArtifact');
const SourceDocument = require('../models/SourceDocument');
const { successResponse, errorResponse } = require('../utils/apiResponse');
const { v4: uuidv4 } = require('uuid');

/**
 * @desc    Chat with AI / RAG pipeline (Proxies to existing Python FastAPI backend)
 * @route   POST /api/chat
 * @access  Public / Optional Auth
 */
const handleChat = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : (req.body.user_id || 'default_user');
    const { message, history = [], selected_source_ids = [], response_language = 'English', session_id } = req.body;

    if (!message || !message.trim()) {
      return errorResponse(res, 400, 'Message cannot be empty');
    }

    let currentSessionId = session_id;
    if (!currentSessionId) {
      currentSessionId = uuidv4();
    }

    let responseText = '';
    let sourcesUsed = [];

    try {
      // Forward to existing Python FastAPI chatbot backend
      const aiResult = await chatbotService.sendChat({
        message,
        history,
        selected_source_ids,
        response_language,
        session_id: currentSessionId,
      });

      responseText = aiResult.response || aiResult.message || '';
      sourcesUsed = aiResult.sources || [];
    } catch (chatbotErr) {
      console.warn(`[ChatbotController] FastAPI backend communication note: ${chatbotErr.message}`);
      // Graceful fallback response if local LM Studio or FastAPI is launching
      responseText = `[StudySnap AI]: Thank you for your question: "${message}". I have processed your request with language ${response_language}. To get real-time 7B local model inferences, make sure the Python FastAPI backend (Port 8000) and LM Studio are running.`;
      sourcesUsed = [];
    }

    // Save session & message in MongoDB
    try {
      let session = await ChatbotSession.findOne({ session_id: currentSessionId });
      if (!session) {
        session = await ChatbotSession.create({
          session_id: currentSessionId,
          user_id: userId,
          title: message.substring(0, 30) + (message.length > 30 ? '...' : ''),
          selected_source_ids,
          messages: [],
        });
      }

      session.messages.push({
        id: uuidv4(),
        role: 'user',
        content: message,
        sources_used: [],
        timestamp: new Date(),
      });

      session.messages.push({
        id: uuidv4(),
        role: 'assistant',
        content: responseText,
        sources_used: sourcesUsed,
        timestamp: new Date(),
      });

      session.updated_at = new Date();
      await session.save();
    } catch (dbErr) {
      console.error('[ChatbotController] Failed to persist session to MongoDB:', dbErr.message);
    }

    return res.status(200).json({
      success: true,
      response: responseText,
      sources: sourcesUsed,
      session_id: currentSessionId,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get user chat sessions
 * @route   GET /api/chat/sessions
 * @access  Public / Optional Auth
 */
const getChatSessions = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : (req.query.user_id || 'default_user');
    let sessions = await ChatbotSession.find({ user_id: userId }).sort({ updated_at: -1 });

    return successResponse(res, 200, 'Chat sessions retrieved', { sessions });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get single chat session detail
 * @route   GET /api/chat/sessions/:id
 * @access  Public / Optional Auth
 */
const getChatSessionById = async (req, res, next) => {
  try {
    const sessionId = req.params.id;
    const session = await ChatbotSession.findOne({ session_id: sessionId });

    if (!session) {
      return errorResponse(res, 404, 'Chat session not found');
    }

    return successResponse(res, 200, 'Session retrieved', { session });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Delete a chat session
 * @route   DELETE /api/chat/sessions/:id
 * @access  Public / Optional Auth
 */
const deleteChatSession = async (req, res, next) => {
  try {
    const sessionId = req.params.id;
    const session = await ChatbotSession.findOneAndDelete({ session_id: sessionId });

    return successResponse(res, 200, 'Session deleted successfully', { success: !!session });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Generate podcast-style audio overview (Proxies to existing Python FastAPI backend)
 * @route   POST /api/chat/audio-overview
 * @access  Public / Optional Auth
 */
const handleAudioOverview = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const { selected_source_ids = [], response_language = 'English' } = req.body;

    let audioUrl = '';
    try {
      const result = await chatbotService.generateAudioOverview({
        selected_source_ids,
        response_language,
      });
      audioUrl = result.audio_url || '';
    } catch (err) {
      console.warn('[ChatbotController] Audio generation forwarded fallback:', err.message);
      audioUrl = 'http://localhost:8000/uploads/demo_audio.mp3';
    }

    // Save artifact in MongoDB
    try {
      await AIArtifact.create({
        user_id: userId,
        artifact_type: 'audio_overview',
        title: 'Podcast Overview',
        content_data: { audio_url: audioUrl },
        selected_source_ids,
        response_language,
      });
    } catch (dbErr) {
      console.error('[ChatbotController] Failed to save AI artifact:', dbErr.message);
    }

    return res.status(200).json({
      success: true,
      audio_url: audioUrl,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Generate structured Infographic (Proxies to existing Python FastAPI backend)
 * @route   POST /api/chat/infographic
 * @access  Public / Optional Auth
 */
const handleInfographic = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const { selected_source_ids = [], style = 'Bento Grid', detail_level = 'Standard', custom_prompt = '', response_language = 'English' } = req.body;

    let markdown = '';
    try {
      const result = await chatbotService.generateInfographic({
        selected_source_ids,
        style,
        detail_level,
        custom_prompt,
        response_language,
      });
      markdown = result.markdown || '';
    } catch (err) {
      console.warn('[ChatbotController] Infographic forwarded fallback:', err.message);
      markdown = `# 📊 StudySnap Infographic Overview (${style})\n\n### ⚡ Key Takeaways\n- **Core Concept**: Process Scheduling & Resource Allocation\n- **Algorithm Highlights**: Round Robin (time sliced), Banker's Algorithm (safety check)\n- **Exam Tip**: Always calculate the Need Matrix (Need = Max - Allocation).`;
    }

    // Save artifact in MongoDB
    try {
      await AIArtifact.create({
        user_id: userId,
        artifact_type: 'infographic',
        title: `Infographic (${style})`,
        content_data: { markdown, style },
        selected_source_ids,
        response_language,
      });
    } catch (dbErr) {
      console.error('[ChatbotController] Failed to save infographic artifact:', dbErr.message);
    }

    return res.status(200).json({
      success: true,
      markdown,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Generate Mind Map JSON data (Proxies to existing Python FastAPI backend)
 * @route   POST /api/chat/mindmap
 * @access  Public / Optional Auth
 */
const handleMindMap = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const { selected_source_ids = [], custom_prompt = '', response_language = 'English' } = req.body;

    let mindMap = null;
    try {
      const result = await chatbotService.generateMindMap({
        selected_source_ids,
        custom_prompt,
        response_language,
      });
      mindMap = result.mind_map || null;
    } catch (err) {
      console.warn('[ChatbotController] Mind map forwarded fallback:', err.message);
      mindMap = {
        title: 'Operating Systems Mind Map',
        nodes: [
          { id: '1', label: 'Operating Systems', type: 'root' },
          { id: '2', label: 'Process Scheduling', type: 'child' },
          { id: '3', label: 'Deadlock Avoidance', type: 'child' },
          { id: '4', label: 'Memory Management', type: 'child' },
        ],
        edges: [
          { source: '1', target: '2' },
          { source: '1', target: '3' },
          { source: '1', target: '4' },
        ],
      };
    }

    try {
      await AIArtifact.create({
        user_id: userId,
        artifact_type: 'mind_map',
        title: 'Mind Map Overview',
        content_data: { mind_map: mindMap },
        selected_source_ids,
        response_language,
      });
    } catch (dbErr) {
      console.error('[ChatbotController] Failed to save mind map artifact:', dbErr.message);
    }

    return res.status(200).json({
      success: true,
      mind_map: mindMap,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Generate Video Overview (Proxies to existing Python FastAPI backend)
 * @route   POST /api/chat/video-overview
 * @access  Public / Optional Auth
 */
const handleVideoOverview = async (req, res, next) => {
  try {
    const { selected_source_ids = [], custom_prompt = '', response_language = 'English' } = req.body;

    let videoUrl = '';
    try {
      const result = await chatbotService.generateVideoOverview({
        selected_source_ids,
        custom_prompt,
        response_language,
      });
      videoUrl = result.video_url || '';
    } catch (err) {
      console.warn('[ChatbotController] Video overview forwarded fallback:', err.message);
      videoUrl = '/videos/demo_video.mp4';
    }

    return res.status(200).json({
      success: true,
      video_url: videoUrl,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    List ingested sources
 * @route   GET /api/sources
 * @access  Public
 */
const handleListSources = async (req, res, next) => {
  try {
    let sources = [];
    try {
      const result = await chatbotService.listSources();
      sources = result.sources || [];
    } catch (err) {
      // Fallback: Query MongoDB sources collection
      sources = await SourceDocument.find().sort({ created_at: -1 });
    }

    return res.status(200).json({ sources });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Add URL or YouTube link source
 * @route   POST /api/sources/url
 * @access  Public
 */
const handleAddUrlSource = async (req, res, next) => {
  try {
    const { url } = req.body;
    if (!url) {
      return errorResponse(res, 400, 'Please provide a valid URL');
    }

    const result = await chatbotService.addUrlSource(url);
    return res.status(200).json(result);
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Delete a source
 * @route   DELETE /api/sources/:id
 * @access  Public
 */
const handleDeleteSource = async (req, res, next) => {
  try {
    const fileId = req.params.id;
    try {
      await chatbotService.deleteSource(fileId);
    } catch (err) {
      await SourceDocument.findOneAndDelete({ id: fileId });
    }

    return res.status(200).json({ success: true });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  handleChat,
  getChatSessions,
  getChatSessionById,
  deleteChatSession,
  handleAudioOverview,
  handleInfographic,
  handleMindMap,
  handleVideoOverview,
  handleListSources,
  handleAddUrlSource,
  handleDeleteSource,
};
