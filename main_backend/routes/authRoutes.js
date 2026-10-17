const express = require('express');
const router = express.Router();
const { register, login, getMe, demoLogin } = require('../controllers/authController');
const { protect } = require('../middleware/authMiddleware');

// Public authentication routes
router.post('/register', register);
router.post('/login', login);
router.post('/demo-login', demoLogin);

// Protected authentication routes
router.get('/me', protect, getMe);

module.exports = router;
