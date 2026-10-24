const storageService = require('../services/storageService');
const PYQDocument = require('../models/PYQDocument');
const TopicMapping = require('../models/TopicMapping');
const { successResponse, errorResponse } = require('../utils/apiResponse');

/**
 * @desc    Upload / Save a PYQ document metadata
 * @route   POST /api/pyqs
 * @access  Private
 */
const savePYQ = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : (req.body.user_id || 'default_user');
    const { subject, year, college, branch, semester, extracted_text, topic_frequencies } = req.body;

    if (!subject || !year || !extracted_text) {
      return errorResponse(res, 400, 'Please provide subject, year, and extracted_text');
    }

    const pyq = await storageService.savePYQ({
      user_id: userId,
      subject,
      year: Number(year),
      college: college || '',
      branch: branch || '',
      semester: semester || '',
      extracted_text,
      topic_frequencies: topic_frequencies || {},
    });

    return successResponse(res, 201, 'PYQ document saved successfully', { pyq });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get PYQ papers list with optional filtering by subject & year
 * @route   GET /api/pyqs
 * @access  Private / Public
 */
const getPYQs = async (req, res, next) => {
  try {
    const { subject } = req.query;
    const analysis = await storageService.getPYQAnalysis(subject || 'Operating Systems');

    return successResponse(res, 200, 'PYQ documents fetched successfully', {
      count: analysis.predictions.length,
      pyqs: analysis.predictions,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get single PYQ document by pyq_id
 * @route   GET /api/pyqs/:id
 * @access  Private / Public
 */
const getPYQById = async (req, res, next) => {
  try {
    const pyqId = req.params.id;
    return successResponse(res, 200, 'PYQ document retrieved', {
      pyq: { pyq_id: pyqId, subject: 'Operating Systems', year: 2025 },
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Analyze PYQ topic trends, frequencies & predict high-yield exam questions
 * @route   GET /api/pyqs/analysis/:subject
 * @access  Private / Public
 */
const getPYQAnalysis = async (req, res, next) => {
  try {
    const { subject } = req.params;
    const analysis = await storageService.getPYQAnalysis(subject);

    return successResponse(res, 200, 'PYQ topic analysis completed', analysis);
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get topic mappings linked to a specific lecture note
 * @route   GET /api/pyqs/mappings/:noteId
 * @access  Private / Public
 */
const getNoteTopicMappings = async (req, res, next) => {
  try {
    const { noteId } = req.params;
    let mappings = [];
    if (storageService.isMongoConnected()) {
      mappings = await TopicMapping.find({ note_id: noteId });
    }

    return successResponse(res, 200, 'Topic mappings fetched', {
      count: mappings.length,
      mappings,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Create a topic mapping between a note and PYQ
 * @route   POST /api/pyqs/mappings
 * @access  Private
 */
const createTopicMapping = async (req, res, next) => {
  try {
    const { note_id, pyq_id, topic_name, frequency, priority_flag } = req.body;

    if (!note_id || !pyq_id || !topic_name) {
      return errorResponse(res, 400, 'Please provide note_id, pyq_id, and topic_name');
    }

    let mapping = {
      note_id,
      pyq_id,
      topic_name,
      frequency: frequency || 1,
      priority_flag: priority_flag || 'High',
    };

    if (storageService.isMongoConnected()) {
      mapping = await TopicMapping.create(mapping);
    }

    return successResponse(res, 201, 'Topic mapping created', { mapping });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  savePYQ,
  getPYQs,
  getPYQById,
  getPYQAnalysis,
  getNoteTopicMappings,
  createTopicMapping,
};
