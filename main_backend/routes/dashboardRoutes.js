const express = require('express');
const router = express.Router();
const { getStats, getNotebooks } = require('../controllers/dashboardController');
const { optionalAuth } = require('../middleware/authMiddleware');

// Student dashboard metrics & notebooks
router.get('/stats', optionalAuth, getStats);
router.get('/notebooks', optionalAuth, getNotebooks);

module.exports = router;
