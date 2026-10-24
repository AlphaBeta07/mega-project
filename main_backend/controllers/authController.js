const bcrypt = require('bcryptjs');
const storageService = require('../services/storageService');
const generateToken = require('../utils/generateToken');
const { successResponse, errorResponse } = require('../utils/apiResponse');

/**
 * @desc    Register a new student or lecturer
 * @route   POST /api/auth/register
 * @access  Public
 */
const register = async (req, res, next) => {
  try {
    const { name, email, password, role, college, branch, semester } = req.body;

    if (!name || !email || !password) {
      return errorResponse(res, 400, 'Please provide name, email, and password');
    }

    if (password.length < 6) {
      return errorResponse(res, 400, 'Password must be at least 6 characters long');
    }

    // Check if user already exists
    const cleanEmail = email.toLowerCase().trim();
    const userExists = await storageService.findUserByEmail(cleanEmail);
    if (userExists) {
      return errorResponse(res, 400, 'An account with this email address already exists');
    }

    // Hash password using bcrypt
    const salt = await bcrypt.genSalt(10);
    const password_hash = await bcrypt.hash(password, salt);

    // Create user via storageService
    const user = await storageService.createUser({
      name: name.trim(),
      email: cleanEmail,
      password_hash,
      role: role || 'student',
      college: college || 'DKTE',
      branch: branch || 'Computer Science',
      semester: semester || 'Sem 5',
    });

    // Generate JWT token
    const token = generateToken({
      userId: user.user_id,
      email: user.email,
      role: user.role,
      name: user.name,
      college: user.college,
    });

    return successResponse(res, 201, 'User registered successfully', {
      user: {
        user_id: user.user_id,
        name: user.name,
        email: user.email,
        role: user.role,
        college: user.college,
        branch: user.branch,
        semester: user.semester,
        created_at: user.created_at,
      },
      token,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Authenticate user & get token (Login)
 * @route   POST /api/auth/login
 * @access  Public
 */
const login = async (req, res, next) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return errorResponse(res, 400, 'Please provide email and password');
    }

    const cleanEmail = email.toLowerCase().trim();
    const user = await storageService.findUserByEmail(cleanEmail);
    if (!user) {
      return errorResponse(res, 401, 'Invalid email or password credentials');
    }

    // Verify password with bcrypt
    const isMatch = await bcrypt.compare(password, user.password_hash);
    if (!isMatch) {
      return errorResponse(res, 401, 'Invalid email or password credentials');
    }

    // Generate JWT token
    const token = generateToken({
      userId: user.user_id,
      email: user.email,
      role: user.role,
      name: user.name,
      college: user.college,
    });

    return successResponse(res, 200, 'Login successful', {
      user: {
        user_id: user.user_id,
        name: user.name,
        email: user.email,
        role: user.role,
        college: user.college,
        branch: user.branch,
        semester: user.semester,
      },
      token,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get current logged in user profile
 * @route   GET /api/auth/me
 * @access  Private (Protected by JWT)
 */
const getMe = async (req, res, next) => {
  try {
    const user = req.user;
    return successResponse(res, 200, 'User profile retrieved', {
      user: {
        user_id: user.user_id,
        name: user.name,
        email: user.email,
        role: user.role,
        college: user.college,
        branch: user.branch,
        semester: user.semester,
        created_at: user.created_at,
      },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Quick demo login for instant student access
 * @route   POST /api/auth/demo-login
 * @access  Public
 */
const demoLogin = async (req, res, next) => {
  try {
    const demoEmail = 'rohan.sharma@iitb.ac.in';
    let user = await storageService.findUserByEmail(demoEmail);

    if (!user) {
      const salt = await bcrypt.genSalt(10);
      const password_hash = await bcrypt.hash('studysnap123', salt);
      user = await storageService.createUser({
        name: 'Rohan Sharma',
        email: demoEmail,
        password_hash,
        role: 'student',
        college: 'DKTE',
        branch: 'Computer Science',
        semester: 'Sem 5',
      });
    }

    const token = generateToken({
      userId: user.user_id,
      email: user.email,
      role: user.role,
      name: user.name,
      college: user.college,
    });

    return successResponse(res, 200, 'Demo student login successful', {
      user: {
        user_id: user.user_id,
        name: user.name,
        email: user.email,
        role: user.role,
        college: user.college,
        branch: user.branch,
        semester: user.semester,
      },
      token,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  register,
  login,
  getMe,
  demoLogin,
};
