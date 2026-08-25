const User = require('../models/User');
const UserSettings = require('../models/UserSettings');
const { successResponse, errorResponse } = require('../utils/apiResponse');

/**
 * @desc    Get user profile details
 * @route   GET /api/users/profile
 * @access  Private
 */
const getProfile = async (req, res, next) => {
  try {
    const user = await User.findOne({ user_id: req.user.user_id });
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
    const { name, college, branch, semester, avatar } = req.body;

    const user = await User.findOne({ user_id: req.user.user_id });
    if (!user) {
      return errorResponse(res, 404, 'User not found');
    }

    if (name) user.name = name;
    if (college) user.college = college;
    if (branch) user.branch = branch;
    if (semester) user.semester = semester;
    if (avatar) user.avatar = avatar;

    await user.save();

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
    let settings = await UserSettings.findOne({ user_id: userId });

    if (!settings) {
      settings = await UserSettings.create({
        user_id: userId,
        response_language: 'English',
        research_mode: 'Fast Research',
        theme: 'dark',
      });
    }

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

    const updateData = { updated_at: new Date() };
    if (response_language) updateData.response_language = response_language;
    if (research_mode) updateData.research_mode = research_mode;
    if (theme) updateData.theme = theme;

    const settings = await UserSettings.findOneAndUpdate(
      { user_id: userId },
      { $set: updateData },
      { new: true, upsert: true }
    );

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
