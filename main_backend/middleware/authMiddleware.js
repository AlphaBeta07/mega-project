const jwt = require('jsonwebtoken');
const User = require('../models/User');
const { errorResponse } = require('../utils/apiResponse');

/**
 * Authentication Middleware
 * Validates JWT token in Authorization header and attaches user to req.user
 */
const protect = async (req, res, next) => {
  let token;

  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith('Bearer')
  ) {
    try {
      token = req.headers.authorization.split(' ')[1];

      const secret = process.env.JWT_SECRET || 'studysnap_jwt_super_secret_key_2026_change_in_production';
      const decoded = jwt.verify(token, secret);

      // Find user by user_id or email
      let user = null;
      if (decoded.userId) {
        user = await User.findOne({ user_id: decoded.userId });
      } else if (decoded.id) {
        user = await User.findById(decoded.id);
      } else if (decoded.email) {
        user = await User.findOne({ email: decoded.email });
      }

      if (!user) {
        // Fallback: construct user from JWT payload if user record was deleted or demo token
        req.user = {
          user_id: decoded.userId || decoded.id || 'default_user',
          name: decoded.name || 'Student User',
          email: decoded.email || 'student@studysnap.ai',
          role: decoded.role || 'student',
          college: decoded.college || 'IIT Bombay',
          branch: decoded.branch || 'Computer Science',
          semester: decoded.semester || 'Sem 5',
        };
      } else {
        req.user = user;
      }

      return next();
    } catch (error) {
      console.error('[Auth Middleware] Token verification failed:', error.message);
      if (error.name === 'TokenExpiredError') {
        return errorResponse(res, 401, 'Authentication token has expired. Please log in again.');
      }
      return errorResponse(res, 401, 'Not authorized, invalid token.');
    }
  }

  if (!token) {
    return errorResponse(res, 401, 'Not authorized, no Bearer token provided.');
  }
};

/**
 * Optional Authentication Middleware
 * Attaches user if token is present, but doesn't block unauthenticated requests
 */
const optionalAuth = async (req, res, next) => {
  if (
    req.headers.authorization &&
    req.headers.authorization.startsWith('Bearer')
  ) {
    try {
      const token = req.headers.authorization.split(' ')[1];
      const secret = process.env.JWT_SECRET || 'studysnap_jwt_super_secret_key_2026_change_in_production';
      const decoded = jwt.verify(token, secret);

      let user = await User.findOne({ user_id: decoded.userId });
      if (user) {
        req.user = user;
      } else {
        req.user = {
          user_id: decoded.userId || 'default_user',
          name: decoded.name || 'Student User',
          email: decoded.email,
          role: decoded.role || 'student',
        };
      }
    } catch (err) {
      req.user = null;
    }
  } else {
    req.user = null;
  }
  next();
};

/**
 * Role Authorization Middleware
 */
const authorize = (...roles) => {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return errorResponse(
        res,
        403,
        `User role '${req.user ? req.user.role : 'guest'}' is not authorized to access this resource`
      );
    }
    next();
  };
};

module.exports = {
  protect,
  optionalAuth,
  authorize,
};
