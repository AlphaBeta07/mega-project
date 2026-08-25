const express = require('express');
const router = express.Router();
const {
  getPublicNotes,
  getUserNotes,
  getNoteById,
  createNote,
  updateNote,
  deleteNote,
  toggleLikeNote,
} = require('../controllers/notesController');
const { protect, optionalAuth } = require('../middleware/authMiddleware');

// Public notes discovery route for LandingPage
router.get('/public', getPublicNotes);

// Notes CRUD routes
router.route('/')
  .get(protect, getUserNotes)
  .post(optionalAuth, createNote);

router.route('/:id')
  .get(optionalAuth, getNoteById)
  .put(protect, updateNote)
  .delete(protect, deleteNote);

// Interaction route
router.post('/:id/like', optionalAuth, toggleLikeNote);

module.exports = router;
