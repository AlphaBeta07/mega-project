const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');

const userSchema = new mongoose.Schema(
  {
    user_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    name: {
      type: String,
      required: [true, 'Please provide a name'],
      trim: true,
    },
    email: {
      type: String,
      required: [true, 'Please provide an email address'],
      unique: true,
      lowercase: true,
      trim: true,
      match: [/^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/, 'Please provide a valid email'],
      index: true,
    },
    password_hash: {
      type: String,
      required: [true, 'Please provide a password hash'],
    },
    role: {
      type: String,
      enum: ['student', 'lecturer', 'admin'],
      default: 'student',
    },
    college: {
      type: String,
      default: 'IIT Bombay',
    },
    branch: {
      type: String,
      default: 'Computer Science',
    },
    semester: {
      type: String,
      default: 'Sem 5',
    },
    avatar: {
      type: String,
      default: '',
    },
    auth_token: {
      type: String,
      default: null,
    },
    created_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'users',
    timestamps: false,
    toJSON: {
      transform: (doc, ret) => {
        delete ret.password_hash;
        delete ret.__v;
        return ret;
      },
    },
  }
);

// Method to compare entered password with bcrypt hash
userSchema.methods.matchPassword = async function (enteredPassword) {
  return await bcrypt.compare(enteredPassword, this.password_hash);
};

// Static helper to hash password
userSchema.statics.hashPassword = async function (password) {
  const salt = await bcrypt.genSalt(10);
  return await bcrypt.hash(password, salt);
};

module.exports = mongoose.model('User', userSchema);
