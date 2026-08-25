const AudioFile = require('../models/AudioFile');
const Transcription = require('../models/Transcription');
const { successResponse, errorResponse } = require('../utils/apiResponse');

/**
 * @desc    Save metadata for an uploaded lecture audio file
 * @route   POST /api/audio
 * @access  Private
 */
const saveAudioFile = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : (req.body.user_id || 'default_user');
    const { file_name, mime_type, file_size, storage_url } = req.body;

    if (!file_name || !file_size || !storage_url) {
      return errorResponse(res, 400, 'Please provide file_name, file_size, and storage_url');
    }

    const audioFile = await AudioFile.create({
      user_id: userId,
      file_name,
      mime_type: mime_type || 'audio/mp3',
      file_size: Number(file_size),
      storage_url,
    });

    return successResponse(res, 201, 'Audio file record created successfully', { audioFile });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get all audio files for the logged in user
 * @route   GET /api/audio
 * @access  Private
 */
const getUserAudioFiles = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : 'default_user';
    const files = await AudioFile.find({ user_id: userId }).sort({ upload_time: -1 });

    return successResponse(res, 200, 'User audio files retrieved', {
      count: files.length,
      files,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Delete an audio file record
 * @route   DELETE /api/audio/:id
 * @access  Private
 */
const deleteAudioFile = async (req, res, next) => {
  try {
    const fileId = req.params.id;
    const file = await AudioFile.findOneAndDelete({ file_id: fileId });

    if (!file) {
      return errorResponse(res, 404, 'Audio file record not found');
    }

    return successResponse(res, 200, 'Audio file record deleted', { file_id: fileId });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get transcription for an audio file
 * @route   GET /api/audio/:id/transcription
 * @access  Private
 */
const getTranscription = async (req, res, next) => {
  try {
    const fileId = req.params.id;
    const transcription = await Transcription.findOne({ file_id: fileId });

    if (!transcription) {
      return errorResponse(res, 404, 'No transcription found for this audio file');
    }

    return successResponse(res, 200, 'Transcription retrieved', { transcription });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Save transcription for an audio file
 * @route   POST /api/audio/transcription
 * @access  Private
 */
const saveTranscription = async (req, res, next) => {
  try {
    const { file_id, raw_text, language, wer_score } = req.body;

    if (!file_id || !raw_text) {
      return errorResponse(res, 400, 'Please provide file_id and raw_text');
    }

    const transcription = await Transcription.create({
      file_id,
      raw_text,
      language: language || 'English',
      wer_score: wer_score || null,
    });

    return successResponse(res, 201, 'Transcription saved successfully', { transcription });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  saveAudioFile,
  getUserAudioFiles,
  deleteAudioFile,
  getTranscription,
  saveTranscription,
};
