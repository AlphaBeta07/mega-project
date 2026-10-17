const jwt = require('jsonwebtoken');

/**
 * Generate a signed JWT token for a user
 * @param {Object} payload - { userId, email, role }
 * @returns {String} JWT token
 */
const generateToken = (payload) => {
  const secret = process.env.JWT_SECRET || 'studysnap_jwt_super_secret_key_2026_change_in_production';
  const expiresIn = process.env.JWT_EXPIRES_IN || '7d';

  return jwt.sign(payload, secret, {
    expiresIn,
  });
};

module.exports = generateToken;
