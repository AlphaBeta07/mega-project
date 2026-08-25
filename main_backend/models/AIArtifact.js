const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const aiArtifactSchema = new mongoose.Schema(
  {
    id: {
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
    artifact_type: {
      type: String,
      enum: ['audio_overview', 'infographic', 'mind_map', 'video_overview', 'flashcards', 'quiz', 'report'],
      required: true,
      index: true,
    },
    title: {
      type: String,
      required: true,
    },
    content_data: {
      type: mongoose.Schema.Types.Mixed,
      required: true,
    },
    selected_source_ids: {
      type: [String],
      default: [],
    },
    response_language: {
      type: String,
      default: 'English',
    },
    created_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'ai_artifacts',
    timestamps: false,
  }
);

module.exports = mongoose.model('AIArtifact', aiArtifactSchema);
