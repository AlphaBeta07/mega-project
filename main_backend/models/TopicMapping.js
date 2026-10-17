const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const topicMappingSchema = new mongoose.Schema(
  {
    mapping_id: {
      type: String,
      default: () => uuidv4(),
      unique: true,
      index: true,
    },
    note_id: {
      type: String,
      required: [true, 'Note ID is required'],
      index: true,
    },
    pyq_id: {
      type: String,
      required: [true, 'PYQ ID is required'],
      index: true,
    },
    topic_name: {
      type: String,
      required: [true, 'Topic name is required'],
    },
    frequency: {
      type: Number,
      default: 1,
    },
    priority_flag: {
      type: String,
      enum: ['High', 'Medium', 'Low'],
      default: 'High',
    },
    created_at: {
      type: Date,
      default: Date.now,
    },
  },
  {
    collection: 'topic_mappings',
    timestamps: false,
  }
);

module.exports = mongoose.model('TopicMapping', topicMappingSchema);
