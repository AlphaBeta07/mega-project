const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const noteSchema = new mongoose.Schema(
  {
    note_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    user_id: {
      type: String,
      required: [true, 'Owner user_id is required'],
      index: true,
    },
    code: {
      type: String,
      default: 'CS101',
    },
    subject: {
      type: String,
      required: [true, 'Subject is required'],
      index: true,
    },
    chapter: {
      type: String,
      required: [true, 'Chapter title is required'],
    },
    title: {
      type: String,
      default: function () {
        return `${this.subject} — ${this.chapter}`;
      },
    },
    description: {
      type: String,
      default: '',
    },
    author: {
      type: String,
      default: 'StudySnap User',
    },
    college: {
      type: String,
      default: 'All Colleges',
    },
    branch: {
      type: String,
      default: 'All Branches',
    },
    semester: {
      type: String,
      default: 'All Semesters',
    },
    note_type: {
      type: String,
      enum: ['Lecture Notes', 'PYQ & Solutions', 'Formula Sheet', 'Lab Manual'],
      default: 'Lecture Notes',
    },
    transcript_text: {
      type: String,
      default: '',
    },
    formatted_content: {
      type: String,
      required: [true, 'Formatted academic content in Markdown is required'],
    },
    summary: {
      type: String,
      default: '',
    },
    keywords: {
      type: [String],
      default: [],
    },
    duration: {
      type: Number,
      default: 0.0,
    },
    audio_file_id: {
      type: String,
      default: null,
    },
    likes_count: {
      type: Number,
      default: 0,
    },
    liked_by: {
      type: [String],
      default: [],
    },
    read_time: {
      type: String,
      default: '5 min read',
    },
    full_content: {
      overview: { type: String, default: '' },
      key_topics: { type: [String], default: [] },
      pyqs: [
        {
          question: { type: String, default: '' },
          year: { type: String, default: '' },
          solution: { type: String, default: '' },
        },
      ],
      formulas: { type: [String], default: [] },
    },
    is_public: {
      type: Boolean,
      default: true,
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
    collection: 'notes',
    timestamps: false,
  }
);

// Enable full text search across subject, chapter, formatted_content, summary, keywords
noteSchema.index({
  subject: 'text',
  chapter: 'text',
  title: 'text',
  description: 'text',
  formatted_content: 'text',
  summary: 'text',
  keywords: 'text',
});

module.exports = mongoose.model('Note', noteSchema);
