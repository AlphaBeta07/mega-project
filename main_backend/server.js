const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const morgan = require('morgan');
const path = require('path');

// Load environment variables
dotenv.config();

// Connect to MongoDB Database
const connectDB = require('./config/db');
connectDB();

const app = express();

// ==========================================
// 1. CORS CONFIGURATION
// ==========================================
const allowedOrigins = (process.env.FRONTEND_URL || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

const defaultOrigins = [
  'http://localhost:5173',
  'http://127.0.0.1:5173',
  'http://localhost:5174',
  'http://127.0.0.1:5174',
  'http://localhost:3000',
];

const corsOptions = {
  origin: (origin, callback) => {
    // Allow requests with no origin (like mobile apps, curl, postman)
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin) || defaultOrigins.includes(origin) || process.env.NODE_ENV !== 'production') {
      return callback(null, true);
    }
    return callback(null, true); // Permissive in development
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
};

app.use(cors(corsOptions));

// ==========================================
// 2. BODY PARSING & LOGGING MIDDLEWARE
// ==========================================
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

if (process.env.NODE_ENV !== 'production') {
  app.use(morgan('dev'));
}

// ==========================================
// 3. HEALTH, ROOT & DIRECTORY ENDPOINTS
// ==========================================
app.get('/favicon.ico', (req, res) => res.status(204).end());

app.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    name: 'StudySnap AI — Main Backend API Gateway',
    version: '1.0.0',
    status: 'online',
    timestamp: new Date().toISOString(),
    api_directory: '/api',
    health_check: '/api/health',
  });
});

app.get('/api', (req, res) => {
  res.status(200).json({
    success: true,
    message: 'Welcome to StudySnap AI Main Backend API Directory',
    endpoints: {
      health: 'GET /api/health',
      auth: {
        register: 'POST /api/auth/register',
        login: 'POST /api/auth/login',
        demo_login: 'POST /api/auth/demo-login',
        me: 'GET /api/auth/me (Protected)',
      },
      notes: {
        public_notes: 'GET /api/notes/public (Supports filters: college, branch, semester, subject, note_type, search, sort_by)',
        user_notes: 'GET /api/notes (Protected)',
        create_note: 'POST /api/notes (Protected)',
        get_note: 'GET /api/notes/:id',
        update_note: 'PUT /api/notes/:id (Protected)',
        delete_note: 'DELETE /api/notes/:id (Protected)',
        like_note: 'POST /api/notes/:id/like',
      },
      dashboard: {
        stats: 'GET /api/dashboard/stats',
        notebooks: 'GET /api/dashboard/notebooks',
      },
      pyqs: {
        list: 'GET /api/pyqs',
        save: 'POST /api/pyqs (Protected)',
        analysis: 'GET /api/pyqs/analysis/:subject',
        mappings: 'GET /api/pyqs/mappings/:noteId',
      },
      audio: {
        list: 'GET /api/audio (Protected)',
        save: 'POST /api/audio (Protected)',
        transcription: 'GET /api/audio/:id/transcription',
      },
      users: {
        profile: 'GET /api/users/profile (Protected)',
        settings: 'GET /api/users/settings (Protected)',
      },
      ai_chatbot: {
        chat: 'POST /api/chat',
        sessions: 'GET /api/chat/sessions',
        audio_overview: 'POST /api/audio-overview',
        infographic: 'POST /api/infographic',
        mindmap: 'POST /api/mindmap',
        video_overview: 'POST /api/video-overview',
        sources: 'GET /api/sources',
      },
    },
    timestamp: new Date().toISOString(),
  });
});

app.get('/api/health', (req, res) => {
  const mongoose = require('mongoose');
  const dbState = mongoose.connection.readyState;
  const dbStatusMap = {
    0: 'disconnected',
    1: 'connected',
    2: 'connecting',
    3: 'disconnecting',
  };

  res.status(200).json({
    success: true,
    server: 'healthy',
    database: {
      status: dbStatusMap[dbState] || 'unknown',
      connected: dbState === 1,
      name: mongoose.connection.name || 'studysnap_db',
    },
    chatbot_backend: {
      url: process.env.CHATBOT_BACKEND_URL || 'http://localhost:8000',
    },
    timestamp: new Date().toISOString(),
  });
});

// ==========================================
// 4. API ROUTE MOUNTING
// ==========================================
const authRoutes = require('./routes/authRoutes');
const userRoutes = require('./routes/userRoutes');
const notesRoutes = require('./routes/notesRoutes');
const pyqRoutes = require('./routes/pyqRoutes');
const audioRoutes = require('./routes/audioRoutes');
const dashboardRoutes = require('./routes/dashboardRoutes');
const chatbotRoutes = require('./routes/chatbotRoutes');

app.use('/api/auth', authRoutes);
app.use('/api/users', userRoutes);
app.use('/api/notes', notesRoutes);
app.use('/api/pyqs', pyqRoutes);
app.use('/api/audio', audioRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api', chatbotRoutes); // Handles /api/chat, /api/sources, /api/audio-overview, etc.

// ==========================================
// 5. ERROR HANDLING MIDDLEWARE
// ==========================================
const { notFound, errorHandler } = require('./middleware/errorMiddleware');

app.use(notFound);
app.use(errorHandler);

// ==========================================
// 6. START SERVER
// ==========================================
const PORT = process.env.PORT || 5000;

let server = null;
if (require.main === module) {
  server = app.listen(PORT, () => {
    console.log(`
  ======================================================
  🚀 StudySnap AI — Main Backend Server Running!
  ------------------------------------------------------
  🌐 Port:        ${PORT}
  📍 Environment: ${process.env.NODE_ENV || 'development'}
  🔗 API Base:    http://localhost:${PORT}/api
  🤖 AI Chatbot:  ${process.env.CHATBOT_BACKEND_URL || 'http://localhost:8000'}
  🗄️ Database:    ${process.env.MONGO_URI || 'mongodb://localhost:27017/studysnap_db'}
  ======================================================
    `);
  });
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (err) => {
  console.error(`[StudySnap Main Backend] Unhandled Rejection: ${err.message}`);
});

module.exports = app;

