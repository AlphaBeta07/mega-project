const axios = require('axios');

/**
 * Chatbot Integration Service
 * Communicates directly with the existing Python FastAPI chatbot backend (Port 8000).
 * DOES NOT rebuild, duplicate, or modify the Python chatbot backend.
 */
class ChatbotService {
  constructor() {
    this.baseUrl = process.env.CHATBOT_BACKEND_URL || 'http://localhost:8000';
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 120000, // 2-min timeout for LLM / TTS generation
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Health check with the existing Python backend
   */
  async checkHealth() {
    try {
      const response = await this.client.get('/api/health/db');
      return response.data;
    } catch (error) {
      return {
        status: 'offline',
        connected: false,
        message: `FastAPI chatbot backend at ${this.baseUrl} is currently unreachable.`,
      };
    }
  }

  /**
   * Send chat message through RAG pipeline
   */
  async sendChat({ message, history = [], selected_source_ids = [], response_language = 'English', session_id = null }) {
    try {
      const payload = {
        message,
        history,
        selected_source_ids,
        response_language,
        session_id,
      };
      const response = await this.client.post('/api/chat', payload);
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error in sendChat:', error.response?.data || error.message);
      throw new Error(
        error.response?.data?.detail ||
        error.response?.data?.message ||
        `Chatbot AI service error: ${error.message}`
      );
    }
  }

  /**
   * Fetch chat sessions for user
   */
  async getSessions(userId = 'default_user') {
    try {
      const response = await this.client.get('/api/sessions', {
        params: { user_id: userId },
      });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error fetching sessions:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Fetch single session detail
   */
  async getSessionById(sessionId) {
    try {
      const response = await this.client.get(`/api/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error fetching session detail:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Delete chat session
   */
  async deleteSession(sessionId) {
    try {
      const response = await this.client.delete(`/api/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error deleting session:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Trigger Audio Podcast Synthesis
   */
  async generateAudioOverview({ selected_source_ids = [], response_language = 'English' }) {
    try {
      const response = await this.client.post('/api/audio-overview', {
        selected_source_ids,
        response_language,
      });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error generating audio overview:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Trigger Infographic Generation
   */
  async generateInfographic({ selected_source_ids = [], style = 'Bento Grid', detail_level = 'Standard', custom_prompt = '', response_language = 'English' }) {
    try {
      const response = await this.client.post('/api/infographic', {
        selected_source_ids,
        style,
        detail_level,
        custom_prompt,
        response_language,
      });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error generating infographic:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Trigger Interactive Mind Map Generation
   */
  async generateMindMap({ selected_source_ids = [], custom_prompt = '', response_language = 'English' }) {
    try {
      const response = await this.client.post('/api/mindmap', {
        selected_source_ids,
        custom_prompt,
        response_language,
      });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error generating mind map:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Trigger Video Overview Generation
   */
  async generateVideoOverview({ selected_source_ids = [], custom_prompt = '', response_language = 'English' }) {
    try {
      const response = await this.client.post('/api/video-overview', {
        selected_source_ids,
        custom_prompt,
        response_language,
      });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error generating video overview:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * List all ingested RAG sources
   */
  async listSources() {
    try {
      const response = await this.client.get('/api/sources');
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error listing sources:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Add URL or YouTube link source
   */
  async addUrlSource(url) {
    try {
      const response = await this.client.post('/api/sources/url', { url });
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error adding URL source:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  /**
   * Delete source by ID
   */
  async deleteSource(fileId) {
    try {
      const response = await this.client.delete(`/api/sources/${fileId}`);
      return response.data;
    } catch (error) {
      console.error('[ChatbotService] Error deleting source:', error.message);
      throw new Error(error.response?.data?.detail || error.message);
    }
  }
}

module.exports = new ChatbotService();
