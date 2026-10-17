const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const pyqDocumentSchema = new mongoose.Schema(
  {
    pyq_id: {
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
    subject: {
      type: String,
      required: [true, 'Subject name is required'],
      index: true,
    },
    year: {
      type: Number,
      required: [true, 'Examination year is required'],
      index: true,
    },
    college: {
      type: String,
      default: '',
    },
    branch: {
      type: String,
      default: '',
    },
    semester: {
      type: String,
      default: '',
    },
    extracted_text: {
      type: String,
      required: [true, 'Extracted question paper text is required'],
    },
    topic_frequencies: {
      type: Map,
      of: Number,
      default: {},
    },
    upload_date: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'pyq_documents',
    timestamps: false,
  }
);

module.exports = mongoose.model('PYQDocument', pyqDocumentSchema);
