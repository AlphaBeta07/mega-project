/**
 * StudySnap AI Main Backend — Comprehensive Automated API Test Suite
 */
const http = require('http');
const app = require('./server');

const PORT = 5099; // Test port to prevent collisions
let server;
let authToken = '';
let createdNoteId = '';

// Helper function to make HTTP requests
const request = (options, postData = null) => {
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port: PORT,
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
          ...(options.headers || {}),
        },
      },
      (res) => {
        let body = '';
        res.on('data', (chunk) => (body += chunk));
        res.on('end', () => {
          try {
            const parsed = body ? JSON.parse(body) : {};
            resolve({ status: res.statusCode, headers: res.headers, body: parsed });
          } catch (e) {
            resolve({ status: res.statusCode, headers: res.headers, body });
          }
        });
      }
    );

    req.on('error', reject);

    if (postData) {
      req.write(typeof postData === 'string' ? postData : JSON.stringify(postData));
    }
    req.end();
  });
};

const runTests = async () => {
  console.log('\n======================================================');
  console.log('🧪 Starting StudySnap Main Backend Test Suite');
  console.log('======================================================\n');

  let passed = 0;
  let failed = 0;

  const assert = (condition, testName, details = '') => {
    if (condition) {
      console.log(`  ✓ PASS: ${testName}`);
      passed++;
    } else {
      console.error(`  ✗ FAIL: ${testName} — ${details}`);
      failed++;
    }
  };

  try {
    // 1. Health Check
    const health = await request({ method: 'GET', path: '/api/health' });
    assert(health.status === 200 && health.body.success, 'GET /api/health (Server & DB Health)');

    // 2. Demo Login
    const demoLoginRes = await request({ method: 'POST', path: '/api/auth/demo-login' });
    assert(
      demoLoginRes.status === 200 && demoLoginRes.body.data?.token,
      'POST /api/auth/demo-login (Instant Access)',
      JSON.stringify(demoLoginRes.body)
    );

    // 3. User Registration
    const testEmail = `test_student_${Date.now()}@studysnap.edu`;
    const regRes = await request(
      { method: 'POST', path: '/api/auth/register' },
      {
        name: 'Aarav Gupta',
        email: testEmail,
        password: 'password123',
        college: 'IIT Bombay',
        branch: 'Computer Science',
        semester: 'Sem 5',
      }
    );
    assert(
      regRes.status === 201 && regRes.body.data?.token,
      'POST /api/auth/register (User Registration with bcrypt & JWT)',
      JSON.stringify(regRes.body)
    );

    if (regRes.body.data?.token) {
      authToken = regRes.body.data.token;
    }

    // 4. User Login
    const loginRes = await request(
      { method: 'POST', path: '/api/auth/login' },
      { email: testEmail, password: 'password123' }
    );
    assert(
      loginRes.status === 200 && loginRes.body.data?.token,
      'POST /api/auth/login (Authentication verification)',
      JSON.stringify(loginRes.body)
    );

    // 5. Protected User Profile (/api/auth/me)
    const meRes = await request({ method: 'GET', path: '/api/auth/me' });
    assert(
      meRes.status === 200 && meRes.body.data?.user?.email === testEmail,
      'GET /api/auth/me (JWT Protected route verification)',
      JSON.stringify(meRes.body)
    );

    // 6. Unauthorized access check (without token)
    const savedToken = authToken;
    authToken = '';
    const unauthRes = await request({ method: 'GET', path: '/api/auth/me' });
    assert(unauthRes.status === 401, 'Protected route rejects unauthenticated request with 401 Unauthorized');
    authToken = savedToken;

    // 7. Public Notes Discovery
    const publicNotesRes = await request({ method: 'GET', path: '/api/notes/public' });
    assert(
      publicNotesRes.status === 200 && Array.isArray(publicNotesRes.body.data?.notes),
      'GET /api/notes/public (Public study notes repository with filters)'
    );

    // 8. Create Lecture Note
    const createNoteRes = await request(
      { method: 'POST', path: '/api/notes' },
      {
        subject: 'Operating Systems',
        chapter: 'Virtual Memory & Paging',
        formatted_content: '# Virtual Memory & Paging\n- Page Tables\n- TLB Miss\n- LRU Page Replacement',
        summary: 'Paging mechanisms and address translation.',
        keywords: ['Paging', 'TLB', 'LRU', 'Virtual Memory'],
        duration: 40.0,
      }
    );
    assert(
      createNoteRes.status === 201 && createNoteRes.body.data?.note?.note_id,
      'POST /api/notes (Create lecture note in MongoDB)',
      JSON.stringify(createNoteRes.body)
    );

    if (createNoteRes.body.data?.note?.note_id) {
      createdNoteId = createNoteRes.body.data.note.note_id;
    }

    // 9. Get User Notes
    const userNotesRes = await request({ method: 'GET', path: '/api/notes' });
    assert(
      userNotesRes.status === 200 && userNotesRes.body.data?.notes?.length > 0,
      'GET /api/notes (Fetch authenticated user notes)'
    );

    // 10. Update Lecture Note
    if (createdNoteId) {
      const updateNoteRes = await request(
        { method: 'PUT', path: `/api/notes/${createdNoteId}` },
        { summary: 'Updated summary for virtual memory.' }
      );
      assert(updateNoteRes.status === 200, 'PUT /api/notes/:id (Update note)');

      // 11. Like Lecture Note
      const likeRes = await request({ method: 'POST', path: `/api/notes/${createdNoteId}/like` });
      assert(likeRes.status === 200 && likeRes.body.data?.likes_count >= 1, 'POST /api/notes/:id/like (Toggle like)');
    }

    // 12. Save PYQ Document
    const pyqRes = await request(
      { method: 'POST', path: '/api/pyqs' },
      {
        subject: 'Operating Systems',
        year: 2025,
        extracted_text: 'Q1: Explain Bankers algorithm with safety criteria.\nQ2: Compare FCFS and Round Robin.',
        topic_frequencies: { 'Bankers Algorithm': 3, 'CPU Scheduling': 4 },
      }
    );
    assert(pyqRes.status === 201, 'POST /api/pyqs (Save PYQ document metadata)');

    // 13. PYQ Trend Analysis & Exam Predictions
    const analysisRes = await request({ method: 'GET', path: '/api/pyqs/analysis/Operating%20Systems' });
    assert(
      analysisRes.status === 200 && analysisRes.body.data?.predictions?.length > 0,
      'GET /api/pyqs/analysis/:subject (Topic frequencies & high-yield exam predictions)'
    );

    // 14. Student Dashboard Stats
    const statsRes = await request({ method: 'GET', path: '/api/dashboard/stats' });
    assert(
      statsRes.status === 200 && statsRes.body.data?.stats?.activeNotebooks,
      'GET /api/dashboard/stats (Real-time aggregated metrics)'
    );

    // 15. Active Notebooks
    const notebooksRes = await request({ method: 'GET', path: '/api/dashboard/notebooks' });
    assert(
      notebooksRes.status === 200 && notebooksRes.body.data?.notebooks?.length > 0,
      'GET /api/dashboard/notebooks (Notebooks collection list)'
    );

    // 16. Chatbot Proxy Endpoint
    const chatRes = await request(
      { method: 'POST', path: '/api/chat' },
      {
        message: 'What is Process Control Block in Operating Systems?',
        response_language: 'English',
      }
    );
    assert(
      chatRes.status === 200 && chatRes.body.response,
      'POST /api/chat (Chatbot proxy & session persistence)'
    );

    // 17. User Settings
    const settingsRes = await request(
      { method: 'PUT', path: '/api/users/settings' },
      { response_language: 'Hinglish', theme: 'dark' }
    );
    assert(settingsRes.status === 200, 'PUT /api/users/settings (Update user preferences)');

    console.log('\n======================================================');
    console.log(`📊 Test Summary: ${passed} Passed, ${failed} Failed`);
    console.log('======================================================\n');
  } catch (err) {
    console.error('Test execution error:', err);
  } finally {
    process.exit(failed === 0 ? 0 : 1);
  }
};

// Start test server
server = app.listen(PORT, () => {
  console.log(`Test server running on port ${PORT}`);
  runTests();
});
