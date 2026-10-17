const express = require('express');
const router = express.Router();
const {
  savePYQ,
  getPYQs,
  getPYQById,
  getPYQAnalysis,
  getNoteTopicMappings,
  createTopicMapping,
} = require('../controllers/pyqController');
const { protect, optionalAuth } = require('../middleware/authMiddleware');

// PYQ Documents routes
router.route('/')
  .get(getPYQs)
  .post(optionalAuth, savePYQ);

// PYQ Analysis and trend predictions route
router.get('/analysis/:subject', getPYQAnalysis);

// Specific PYQ document
router.get('/:id', getPYQById);

// Topic Mappings routes
router.get('/mappings/:noteId', getNoteTopicMappings);
router.post('/mappings', protect, createTopicMapping);

module.exports = router;
