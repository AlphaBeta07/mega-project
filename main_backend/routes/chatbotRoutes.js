const express = require('express');
const router = express.Router();
const {
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
} = require('../controllers/chatbotController');
const { optionalAuth, protect } = require('../middleware/authMiddleware');

// Chat and RAG conversational routes
router.post('/chat', optionalAuth, handleChat);
router.get('/chat/sessions', optionalAuth, getChatSessions);
router.get('/chat/sessions/:id', optionalAuth, getChatSessionById);
router.delete('/chat/sessions/:id', optionalAuth, deleteChatSession);

// Direct session alias matching FastAPI
router.get('/sessions', optionalAuth, getChatSessions);
router.get('/sessions/:id', optionalAuth, getChatSessionById);
router.delete('/sessions/:id', optionalAuth, deleteChatSession);

// AI Generation routes
router.post('/audio-overview', optionalAuth, handleAudioOverview);
router.post('/infographic', optionalAuth, handleInfographic);
router.post('/mindmap', optionalAuth, handleMindMap);
router.post('/video-overview', optionalAuth, handleVideoOverview);

// Source ingestion routes
router.get('/sources', handleListSources);
router.post('/sources/url', handleAddUrlSource);
router.delete('/sources/:id', handleDeleteSource);

module.exports = router;
