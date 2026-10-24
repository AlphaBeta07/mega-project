const storageService = require('../services/storageService');
const { successResponse } = require('../utils/apiResponse');

/**
 * @desc    Get aggregated student dashboard metrics and statistics
 * @route   GET /api/dashboard/stats
 * @access  Private / Public
 */
const getStats = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const stats = await storageService.getDashboardStats(userId);

    return successResponse(res, 200, 'Dashboard statistics fetched successfully', {
      stats,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get list of active study notebooks for student dashboard
 * @route   GET /api/dashboard/notebooks
 * @access  Private / Public
 */
const getNotebooks = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const notebooks = await storageService.getDashboardNotebooks(userId);

    return successResponse(res, 200, 'Active notebooks fetched successfully', {
      count: notebooks.length,
      notebooks,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getStats,
  getNotebooks,
};
