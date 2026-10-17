const mongoose = require('mongoose');

const userSettingsSchema = new mongoose.Schema(
  {
    user_id: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    response_language: {
      type: String,
      default: 'English',
    },
    research_mode: {
      type: String,
      default: 'Fast Research',
    },
    theme: {
      type: String,
      default: 'dark',
    },
    updated_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'user_settings',
    timestamps: false,
  }
);

module.exports = mongoose.model('UserSettings', userSettingsSchema);
