const express = require('express');
const router = express.Router();
const {
  getProfile,
  updateProfile,
  getSettings,
  updateSettings,
} = require('../controllers/userController');
const { protect } = require('../middleware/authMiddleware');

// User profile & preferences routes
router.get('/profile', protect, getProfile);
router.put('/profile', protect, updateProfile);
router.get('/settings', protect, getSettings);
router.put('/settings', protect, updateSettings);

module.exports = router;
