const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const chatMessageSchema = new mongoose.Schema(
  {
    id: {
      type: String,
      default: () => uuidv4(),
    },
    role: {
      type: String,
      enum: ['user', 'assistant', 'system'],
      required: true,
    },
    content: {
      type: String,
      required: true,
    },
    sources_used: {
      type: [mongoose.Schema.Types.Mixed],
      default: [],
    },
    timestamp: {
      type: Date,
      default: Date.now,
    },
  },
  { _id: false }
);

const chatbotSessionSchema = new mongoose.Schema(
  {
    session_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    user_id: {
      type: String,
      default: 'default_user',
      index: true,
    },
    title: {
      type: String,
      default: 'New Lecture Doubt Chat',
    },
    messages: {
      type: [chatMessageSchema],
      default: [],
    },
    selected_source_ids: {
      type: [String],
      default: [],
    },
    created_at: {
      type: Date,
      default: Date.now,
    },
    updated_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'chatbot_sessions',
    timestamps: false,
  }
);

module.exports = mongoose.model('ChatbotSession', chatbotSessionSchema);
