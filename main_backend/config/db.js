const mongoose = require('mongoose');

/**
 * Connect to MongoDB Database
 * Supports local MongoDB and MongoDB Atlas clusters.
 */
const connectDB = async () => {
  try {
    const connUri = process.env.MONGO_URI || 'mongodb://localhost:27017/studysnap_db';
    
    // Configure mongoose connection options
    const options = {
      autoIndex: true,
      serverSelectionTimeoutMS: 5000,
    };

    const conn = await mongoose.connect(connUri, options);

    console.log(`[StudySnap Main Backend] MongoDB Connected: ${conn.connection.host}/${conn.connection.name}`);

    // Listen to connection events
    mongoose.connection.on('error', (err) => {
      console.error(`[StudySnap Main Backend] MongoDB Connection Error: ${err.message}`);
    });

    mongoose.connection.on('disconnected', () => {
      console.warn('[StudySnap Main Backend] MongoDB Disconnected. Reconnecting...');
    });

    return conn;
  } catch (error) {
    console.error(`[StudySnap Main Backend] Database Connection Failed: ${error.message}`);
    // Do not crash server in dev mode if DB is temporarily offline
    if (process.env.NODE_ENV === 'production') {
      process.exit(1);
    }
  }
};

module.exports = connectDB;
