const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const audioFileSchema = new mongoose.Schema(
  {
    file_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    user_id: {
      type: String,
      required: [true, 'User ID is required'],
      index: true,
    },
    file_name: {
      type: String,
      required: [true, 'Original audio filename is required'],
    },
    mime_type: {
      type: String,
      default: 'audio/mp3',
    },
    file_size: {
      type: Number,
      required: [true, 'File size in bytes is required'],
    },
    storage_url: {
      type: String,
      required: [true, 'Storage URL is required'],
    },
    upload_time: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'audio_files',
    timestamps: false,
  }
);

module.exports = mongoose.model('AudioFile', audioFileSchema);
