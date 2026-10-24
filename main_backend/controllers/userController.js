const storageService = require('../services/storageService');
const { successResponse, errorResponse } = require('../utils/apiResponse');

/**
 * @desc    Get user profile details
 * @route   GET /api/users/profile
 * @access  Private
 */
const getProfile = async (req, res, next) => {
  try {
    const user = await storageService.findUserById(req.user.user_id);
    if (!user) {
      return errorResponse(res, 404, 'User profile not found');
    }
    return successResponse(res, 200, 'User profile', { user });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Update user profile details
 * @route   PUT /api/users/profile
 * @access  Private
 */
const updateProfile = async (req, res, next) => {
  try {
    const user = await storageService.findUserById(req.user.user_id);
    if (!user) {
      return errorResponse(res, 404, 'User not found');
    }

    const { name, college, branch, semester } = req.body;
    if (name) user.name = name;
    if (college) user.college = college;
    if (branch) user.branch = branch;
    if (semester) user.semester = semester;

    return successResponse(res, 200, 'Profile updated successfully', { user });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get user application settings & preferences
 * @route   GET /api/users/settings
 * @access  Private
 */
const getSettings = async (req, res, next) => {
  try {
    const userId = req.user.user_id;
    const settings = await storageService.getUserSettings(userId);
    return successResponse(res, 200, 'User settings', { settings });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Update user settings
 * @route   PUT /api/users/settings
 * @access  Private
 */
const updateSettings = async (req, res, next) => {
  try {
    const userId = req.user.user_id;
    const { response_language, research_mode, theme } = req.body;

    const settings = await storageService.updateUserSettings(userId, {
      response_language,
      research_mode,
      theme,
    });

    return successResponse(res, 200, 'User settings updated successfully', { settings });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getProfile,
  updateProfile,
  getSettings,
  updateSettings,
};
