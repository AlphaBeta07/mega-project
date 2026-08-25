const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const sourceDocumentSchema = new mongoose.Schema(
  {
    id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    filename: {
      type: String,
      required: [true, 'Filename or URL title is required'],
    },
    type: {
      type: String,
      required: [true, 'Source type is required (pdf, docx, txt, audio, url, youtube)'],
    },
    file_path: {
      type: String,
      default: null,
    },
    url: {
      type: String,
      default: null,
    },
    raw_text: {
      type: String,
      default: null,
    },
    chunk_count: {
      type: Number,
      default: 0,
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
    collection: 'sources',
    timestamps: false,
  }
);

module.exports = mongoose.model('SourceDocument', sourceDocumentSchema);
