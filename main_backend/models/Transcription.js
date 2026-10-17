const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const transcriptionSchema = new mongoose.Schema(
  {
    transcription_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    file_id: {
      type: String,
      required: [true, 'Audio file ID is required'],
      index: true,
    },
    raw_text: {
      type: String,
      required: [true, 'Raw transcription text is required'],
    },
    language: {
      type: String,
      default: 'English',
    },
    wer_score: {
      type: Number,
      default: null,
    },
    processed_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'transcriptions',
    timestamps: false,
  }
);

module.exports = mongoose.model('Transcription', transcriptionSchema);
