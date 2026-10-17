const express = require('express');
const router = express.Router();
const {
  saveAudioFile,
  getUserAudioFiles,
  deleteAudioFile,
  getTranscription,
  saveTranscription,
} = require('../controllers/audioController');
const { protect, optionalAuth } = require('../middleware/authMiddleware');

// Audio files routes
router.route('/')
  .get(protect, getUserAudioFiles)
  .post(optionalAuth, saveAudioFile);

router.delete('/:id', protect, deleteAudioFile);

// Transcriptions routes
router.get('/:id/transcription', getTranscription);
router.post('/transcription', protect, saveTranscription);

module.exports = router;
