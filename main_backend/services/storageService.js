const fs = require('fs');
const path = require('path');
const mongoose = require('mongoose');
const User = require('../models/User');
const Note = require('../models/Note');
const PYQDocument = require('../models/PYQDocument');
const AudioFile = require('../models/AudioFile');

const DATA_DIR = path.join(__dirname, '..', 'data');
const STORE_FILE = path.join(DATA_DIR, 'store.json');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Initial seed notes matching frontend requirements
const DEFAULT_INITIAL_NOTES = [
  {
    note_id: 'note-1',
    user_id: 'usr_rohan_demo',
    code: 'CS302',
    subject: 'Operating Systems',
    chapter: 'Module 1: Process Scheduling & Deadlocks',
    title: 'Operating Systems — Process Scheduling & Deadlocks',
    college: 'DKTE',
    branch: 'Computer Science',
    semester: 'Sem 5',
    note_type: 'Lecture Notes',
    description: "Complete breakdown of CPU scheduling algorithms (FCFS, SJF, Round Robin), Banker's algorithm for deadlock avoidance with resource allocation graph examples.",
    author: 'Rohan Sharma (DKTE)',
    likes_count: 142,
    read_time: '8 min read',
    formatted_content: '# Operating Systems — Process Scheduling & Deadlocks\n\n## Overview\nThis comprehensive guide covers kernel process management, CPU scheduler implementations in Linux, deadlock conditions (Coffman conditions), and Banker algorithm matrices.\n\n## Key Concepts\n- FCFS, SJF, SRTF, and Round Robin Gantt Chart examples\n- Multi-level Queue Scheduling & Feedback Queues\n- 4 Necessary Conditions for Deadlock: Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait\n- Resource Allocation Graph (RAG) reduction and Banker\'s Safety Algorithm\n\n## Essential Formulas\n- Turnaround Time (TAT) = Completion Time - Arrival Time\n- Waiting Time (WT) = Turnaround Time - Burst Time\n- Need[i][j] = Max[i][j] - Allocation[i][j]',
    summary: 'Process scheduling algorithms, Gantt charts, Coffman deadlock conditions, and Banker algorithm calculations.',
    keywords: ['Process', 'Scheduling', 'Deadlock', 'Banker Algorithm', 'Gantt Chart'],
    duration: 45.0,
    full_content: {
      overview: 'This comprehensive guide covers kernel process management, CPU scheduler implementations in Linux, deadlock conditions (Coffman conditions), and Banker algorithm matrices.',
      key_topics: [
        'FCFS, SJF, SRTF, and Round Robin Gantt Chart examples',
        'Multi-level Queue Scheduling & Feedback Queues',
        '4 Necessary Conditions for Deadlock: Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait',
        "Resource Allocation Graph (RAG) reduction and Banker's Safety Algorithm",
      ],
      pyqs: [
        {
          question: 'Consider 5 processes P0 through P4 with Allocation and Max matrices. Calculate the Need matrix and verify if the system is in a safe state.',
          year: 'Endsem 2025',
          solution: 'Need = Max - Allocation. System is in safe state with sequence <P1, P3, P4, P0, P2>.',
        },
        {
          question: 'Differentiate between preemptive and non-preemptive scheduling with time-quantum trade-offs.',
          year: 'Midsem 2024',
          solution: 'Preemptive scheduling interrupts running tasks when higher priority arrives. Quantum too small increases context switch overhead; quantum too large degrades to FCFS.',
        },
      ],
      formulas: [
        'Turnaround Time (TAT) = Completion Time - Arrival Time',
        'Waiting Time (WT) = Turnaround Time - Burst Time',
        'Need[i][j] = Max[i][j] - Allocation[i][j]',
      ],
    },
    is_public: true,
    created_at: new Date('2026-03-10T10:00:00Z'),
  },
  {
    note_id: 'note-2',
    user_id: 'seed_user_2',
    code: 'CS201',
    subject: 'Data Structures',
    chapter: 'Module 3: AVL Tree Rotations & Red-Black Trees',
    title: 'Data Structures — AVL Tree Rotations & Red-Black Trees',
    college: 'PES University',
    branch: 'Computer Science',
    semester: 'Sem 3',
    note_type: 'Formula Sheet',
    description: 'Complete step-by-step balance factor derivations for Single (LL, RR) and Double (LR, RL) rotations with time complexity proofs and code templates.',
    author: 'Priya Verma (PES University)',
    likes_count: 215,
    read_time: '12 min read',
    formatted_content: '# Data Structures — AVL Trees & Balanced BSTs\n\n## Overview\nDetailed visual cheat-sheet and code notes for self-balancing binary search trees, balance factor computation, and Red-Black tree insertion re-coloring rules.\n\n## Key Topics\n- Balance Factor = Height(Left Subtree) - Height(Right Subtree) in {-1, 0, +1}\n- Single Rotations: LL Rotation (Right Rotate), RR Rotation (Left Rotate)\n- Double Rotations: LR Rotation (Left-Right), RL Rotation (Right-Left)\n- Red-Black Tree Properties: Root is black, No adjacent red nodes, Black-height equality',
    summary: 'Self-balancing BSTs, AVL tree single and double rotations, and Red-Black tree coloring rules.',
    keywords: ['AVL Tree', 'BST', 'Rotations', 'Red-Black Tree', 'Balance Factor'],
    duration: 35.0,
    full_content: {
      overview: 'Detailed visual cheat-sheet and code notes for self-balancing binary search trees, balance factor computation, and Red-Black tree insertion re-coloring rules.',
      key_topics: [
        'Balance Factor = Height(Left Subtree) - Height(Right Subtree) ∈ {-1, 0, +1}',
        'Single Rotations: LL Rotation (Right Rotate), RR Rotation (Left Rotate)',
        'Double Rotations: LR Rotation (Left-Right), RL Rotation (Right-Left)',
        'Red-Black Tree Properties: Root is black, No adjacent red nodes, Black-height equality',
      ],
      pyqs: [
        {
          question: 'Insert keys 14, 17, 11, 7, 53, 4, 13 into an empty AVL tree and show rebalancing at each step.',
          year: 'Endsem 2024',
          solution: 'Inserting 4 triggers LL imbalance at node 11. Right rotate at 11 yields balanced tree with root 11.',
        },
      ],
      formulas: [
        'Height of AVL tree h <= 1.44 log2(N + 2)',
        'Search / Insert / Delete Time Complexity = O(log N)',
      ],
    },
    is_public: true,
    created_at: new Date('2026-03-08T14:30:00Z'),
  },
  {
    note_id: 'note-3',
    user_id: 'seed_user_3',
    code: 'EC401',
    subject: 'Digital Signal Processing',
    chapter: 'Module 4: FFT Decimation Algorithm',
    title: 'Digital Signal Processing — Fast Fourier Transform (FFT) Decimation',
    college: 'IIT Delhi',
    branch: 'Electronics & Communication',
    semester: 'Sem 5',
    note_type: 'Lecture Notes',
    description: 'Radix-2 DIT-FFT butterfly computation formulas, twiddle factor properties, and step-by-step 8-point DFT problem with complete signal flow diagram.',
    author: 'Ananya Sen (IIT Delhi)',
    likes_count: 98,
    read_time: '15 min read',
    formatted_content: '# Digital Signal Processing — Fast Fourier Transform (FFT)\n\n## Overview\nComplete mathematical formulation of Cooley-Tukey Radix-2 Decimation-In-Time (DIT) FFT, computational complexity comparison O(N log N) vs O(N^2), and twiddle factor symmetries.',
    summary: 'Radix-2 DIT-FFT butterfly diagrams and twiddle factor periodicity.',
    keywords: ['FFT', 'DFT', 'Twiddle Factor', 'Butterflies', 'Signal Processing'],
    duration: 50.0,
    full_content: {
      overview: 'Complete mathematical formulation of Cooley-Tukey Radix-2 Decimation-In-Time (DIT) FFT, computational complexity comparison O(N log N) vs O(N^2), and twiddle factor symmetries.',
      key_topics: [
        'DFT direct computation: N^2 complex multiplications vs FFT: (N/2) log2 N',
        'Twiddle Factor W_N^k = e^(-j 2 pi k / N)',
        'Symmetry: W_N^(k + N/2) = -W_N^k; Periodicity: W_N^(k + N) = W_N^k',
      ],
      pyqs: [
        {
          question: 'Compute 8-point DFT of sequence x(n) = {1, 2, 1, 0, 0, 0, 0, 0} using Radix-2 DIT-FFT algorithm.',
          year: 'Endsem 2025',
          solution: 'Bit reverse input array to {x(0), x(4), x(2), x(6), x(1), x(5), x(3), x(7)}. Perform 3 stages of butterfly computation.',
        },
      ],
      formulas: [
        'X(k) = sum_{n=0}^{N-1} x(n) * W_N^{n*k}',
      ],
    },
    is_public: true,
    created_at: new Date('2026-03-05T09:15:00Z'),
  },
  {
    note_id: 'note-4',
    user_id: 'seed_user_4',
    code: 'CS504',
    subject: 'Database Systems',
    chapter: 'Module 2: 3NF & BCNF Normalization',
    title: 'Database Management — 3NF & BCNF Normalization Solved PYQs',
    college: 'BITS Pilani',
    branch: 'Computer Science',
    semester: 'Sem 5',
    note_type: 'PYQ & Solutions',
    description: 'Step-by-step resolution of functional dependencies, minimal cover finding, 3NF and BCNF decomposition with lossless join and dependency preservation proofs.',
    author: 'Sneha Reddy (BITS Pilani)',
    likes_count: 180,
    read_time: '10 min read',
    formatted_content: '# Database Management — Normalization & Functional Dependencies\n\nMaster relational schema normalization with practical algorithms to test 2NF, 3NF, BCNF, and dependency preserving decompositions.',
    summary: 'Functional dependencies, 3NF synthesis, and BCNF decomposition.',
    keywords: ['Normalization', '3NF', 'BCNF', 'Functional Dependency', 'DBMS'],
    duration: 40.0,
    full_content: {
      overview: 'Master relational schema normalization with practical algorithms to test 2NF, 3NF, BCNF, and dependency preserving decompositions.',
      key_topics: [
        'Candidate Key determination using Attribute Closure',
        'Minimal Cover / Canonical Cover calculation',
        '3NF Synthesis Algorithm',
        'BCNF Decomposition Algorithm (Checking X -> Y where X is superkey)',
      ],
      pyqs: [
        {
          question: 'Given R(A,B,C,D,E) with F = {A->BC, CD->E, B->D, E->A}. Find candidate keys and decompose R into BCNF.',
          year: 'Endsem 2024',
          solution: 'Candidate keys are {A}, {E}, {BC}. Relation is in 3NF but not BCNF due to B->D. Decomposed into R1(B,D) and R2(A,B,C,E).',
        },
      ],
      formulas: [],
    },
    is_public: true,
    created_at: new Date('2026-03-01T16:45:00Z'),
  },
];

class StorageService {
  constructor() {
    this._initStore();
  }

  _initStore() {
    if (!fs.existsSync(STORE_FILE)) {
      const initialData = {
        users: [
          {
            user_id: 'usr_rohan_demo',
            name: 'Rohan Sharma',
            email: 'rohan.sharma@iitb.ac.in',
            password_hash: '$2a$10$vY3Z6a6v8D5d7tV3QZzMhO.5X5Z7W8P8Q9R0S1T2U3V4W5X6Y7Z8a', // studysnap123
            role: 'student',
            college: 'DKTE',
            branch: 'Computer Science',
            semester: 'Sem 5',
            created_at: new Date().toISOString(),
          },
        ],
        notes: DEFAULT_INITIAL_NOTES,
        stats: {
          activeNotebooks: 4,
          uploadedSources: 14,
          audioSynthesis: 3,
          solvedPYQs: 28,
        },
      };
      fs.writeFileSync(STORE_FILE, JSON.stringify(initialData, null, 2), 'utf-8');
    }
  }

  _readStore() {
    try {
      this._initStore();
      const content = fs.readFileSync(STORE_FILE, 'utf-8');
      return JSON.parse(content);
    } catch (e) {
      console.error('[StorageService] Error reading store file:', e.message);
      return { users: [], notes: DEFAULT_INITIAL_NOTES, stats: {} };
    }
  }

  _writeStore(data) {
    try {
      fs.writeFileSync(STORE_FILE, JSON.stringify(data, null, 2), 'utf-8');
    } catch (e) {
      console.error('[StorageService] Error writing store file:', e.message);
    }
  }

  isMongoConnected() {
    return mongoose.connection.readyState === 1;
  }

  // ==========================================
  // USER METHODS
  // ==========================================
  async findUserByEmail(email) {
    const cleanEmail = (email || '').toLowerCase().trim();
    if (this.isMongoConnected()) {
      try {
        const user = await User.findOne({ email: cleanEmail });
        if (user) return user;
      } catch (err) {
        console.warn('[StorageService] Mongo findUserByEmail failed, checking local store:', err.message);
      }
    }
    const store = this._readStore();
    return store.users.find((u) => u.email.toLowerCase() === cleanEmail) || null;
  }

  async findUserById(userId) {
    if (this.isMongoConnected()) {
      try {
        const user = await User.findOne({ user_id: userId });
        if (user) return user;
      } catch (err) {
        console.warn('[StorageService] Mongo findUserById failed, checking local store:', err.message);
      }
    }
    const store = this._readStore();
    return store.users.find((u) => u.user_id === userId) || null;
  }

  async createUser(userData) {
    if (this.isMongoConnected()) {
      try {
        const user = await User.create(userData);
        // Also keep local store updated
        const store = this._readStore();
        store.users.push(user.toObject ? user.toObject() : userData);
        this._writeStore(store);
        return user;
      } catch (err) {
        console.warn('[StorageService] Mongo createUser failed, saving to local store:', err.message);
      }
    }
    const store = this._readStore();
    const newUser = {
      ...userData,
      user_id: userData.user_id || `usr_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      created_at: new Date().toISOString(),
    };
    store.users.push(newUser);
    this._writeStore(store);
    return newUser;
  }

  // ==========================================
  // NOTE METHODS
  // ==========================================
  async getPublicNotes(filters = {}) {
    if (this.isMongoConnected()) {
      try {
        const count = await Note.countDocuments();
        if (count === 0) {
          await Note.insertMany(DEFAULT_INITIAL_NOTES);
        }

        const query = { is_public: true };
        if (filters.college && filters.college !== 'All Colleges') query.college = filters.college;
        if (filters.branch && filters.branch !== 'All Branches') query.branch = filters.branch;
        if (filters.semester && filters.semester !== 'All Semesters') query.semester = filters.semester;
        if (filters.subject && filters.subject !== 'All Subjects') query.subject = filters.subject;
        if (filters.note_type && filters.note_type !== 'All Types') query.note_type = filters.note_type;

        if (filters.search && filters.search.trim()) {
          const reg = new RegExp(filters.search.trim(), 'i');
          query.$or = [
            { title: reg },
            { subject: reg },
            { chapter: reg },
            { description: reg },
            { author: reg },
            { code: reg },
          ];
        }

        let sortOption = { created_at: -1 };
        if (filters.sort_by === 'Most Liked') sortOption = { likes_count: -1 };
        if (filters.sort_by === 'Title A-Z') sortOption = { title: 1 };

        return await Note.find(query).sort(sortOption);
      } catch (err) {
        console.warn('[StorageService] Mongo getPublicNotes failed, falling back to store:', err.message);
      }
    }

    const store = this._readStore();
    let notes = (store.notes || []).filter((n) => n.is_public !== false);

    if (filters.college && filters.college !== 'All Colleges') {
      notes = notes.filter((n) => n.college === filters.college);
    }
    if (filters.branch && filters.branch !== 'All Branches') {
      notes = notes.filter((n) => n.branch === filters.branch);
    }
    if (filters.semester && filters.semester !== 'All Semesters') {
      notes = notes.filter((n) => n.semester === filters.semester);
    }
    if (filters.subject && filters.subject !== 'All Subjects') {
      notes = notes.filter((n) => n.subject === filters.subject);
    }
    if (filters.note_type && filters.note_type !== 'All Types') {
      notes = notes.filter((n) => n.note_type === filters.note_type);
    }
    if (filters.search && filters.search.trim()) {
      const q = filters.search.trim().toLowerCase();
      notes = notes.filter(
        (n) =>
          (n.title && n.title.toLowerCase().includes(q)) ||
          (n.subject && n.subject.toLowerCase().includes(q)) ||
          (n.chapter && n.chapter.toLowerCase().includes(q)) ||
          (n.description && n.description.toLowerCase().includes(q)) ||
          (n.author && n.author.toLowerCase().includes(q)) ||
          (n.code && n.code.toLowerCase().includes(q))
      );
    }

    if (filters.sort_by === 'Most Liked') {
      notes.sort((a, b) => (b.likes_count || 0) - (a.likes_count || 0));
    } else if (filters.sort_by === 'Title A-Z') {
      notes.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
    } else {
      notes.sort((a, b) => new Date(b.created_at || b.date || 0) - new Date(a.created_at || a.date || 0));
    }

    return notes;
  }

  async getNoteById(noteId) {
    if (this.isMongoConnected()) {
      try {
        const note = await Note.findOne({ note_id: noteId });
        if (note) return note;
      } catch (err) {
        console.warn('[StorageService] Mongo getNoteById failed:', err.message);
      }
    }
    const store = this._readStore();
    return store.notes.find((n) => n.note_id === noteId || n._id === noteId) || null;
  }

  async createNote(noteData) {
    if (this.isMongoConnected()) {
      try {
        const note = await Note.create(noteData);
        const store = this._readStore();
        store.notes.push(note.toObject ? note.toObject() : noteData);
        this._writeStore(store);
        return note;
      } catch (err) {
        console.warn('[StorageService] Mongo createNote failed:', err.message);
      }
    }
    const store = this._readStore();
    const newNote = {
      ...noteData,
      note_id: noteData.note_id || `note_${Date.now()}`,
      created_at: new Date().toISOString(),
      likes_count: noteData.likes_count || 0,
      is_public: noteData.is_public !== undefined ? noteData.is_public : true,
    };
    store.notes.push(newNote);
    this._writeStore(store);
    return newNote;
  }

  async likeNote(noteId) {
    if (this.isMongoConnected()) {
      try {
        const note = await Note.findOneAndUpdate(
          { note_id: noteId },
          { $inc: { likes_count: 1 } },
          { new: true }
        );
        if (note) return note;
      } catch (err) {
        console.warn('[StorageService] Mongo likeNote failed:', err.message);
      }
    }
    const store = this._readStore();
    const note = store.notes.find((n) => n.note_id === noteId);
    if (note) {
      note.likes_count = (note.likes_count || 0) + 1;
      this._writeStore(store);
      return note;
    }
    return null;
  }

  async updateNote(noteId, updateData) {
    if (this.isMongoConnected()) {
      try {
        const note = await Note.findOneAndUpdate(
          { note_id: noteId },
          { $set: updateData },
          { new: true }
        );
        if (note) return note;
      } catch (err) {
        console.warn('[StorageService] Mongo updateNote failed:', err.message);
      }
    }
    const store = this._readStore();
    const note = store.notes.find((n) => n.note_id === noteId);
    if (note) {
      Object.assign(note, updateData);
      this._writeStore(store);
      return note;
    }
    return null;
  }

  async deleteNote(noteId) {
    if (this.isMongoConnected()) {
      try {
        await Note.findOneAndDelete({ note_id: noteId });
      } catch (err) {
        console.warn('[StorageService] Mongo deleteNote failed:', err.message);
      }
    }
    const store = this._readStore();
    store.notes = (store.notes || []).filter((n) => n.note_id !== noteId);
    this._writeStore(store);
    return true;
  }

  // ==========================================
  // PYQ METHODS
  // ==========================================
  async savePYQ(pyqData) {
    if (this.isMongoConnected()) {
      try {
        const doc = await PYQDocument.create(pyqData);
        return doc;
      } catch (err) {
        console.warn('[StorageService] Mongo savePYQ failed:', err.message);
      }
    }
    const store = this._readStore();
    if (!store.pyqs) store.pyqs = [];
    const newPyq = {
      ...pyqData,
      pyq_id: `pyq_${Date.now()}`,
      created_at: new Date().toISOString(),
    };
    store.pyqs.push(newPyq);
    this._writeStore(store);
    return newPyq;
  }

  async getPYQAnalysis(subject) {
    const store = this._readStore();
    const cleanSubj = (subject || '').toLowerCase();
    const relevantPyqs = (store.pyqs || []).filter(
      (p) => (p.subject || '').toLowerCase() === cleanSubj
    );

    const predictions = [
      { topic: 'Bankers Algorithm & Deadlock Avoidance', frequency: 5, probability: '92%', yield_type: 'High Yield' },
      { topic: 'CPU Scheduling (Round Robin & SJF)', frequency: 4, probability: '85%', yield_type: 'High Yield' },
      { topic: 'Virtual Memory & Page Replacement (LRU)', frequency: 4, probability: '82%', yield_type: 'High Yield' },
      { topic: 'Process Synchronization & Semaphores', frequency: 3, probability: '74%', yield_type: 'Medium Yield' },
      { topic: 'Disk Scheduling Algorithms (SCAN, C-LOOK)', frequency: 2, probability: '65%', yield_type: 'Medium Yield' },
    ];

    return {
      subject,
      total_pyqs_analyzed: Math.max(relevantPyqs.length, 6),
      predictions,
      exam_tips: [
        'Practice resource allocation matrices for Banker Algorithm (frequently carries 10+ marks).',
        'Draw clean Gantt charts showing waiting time and turnaround time calculations.',
      ],
    };
  }

  // ==========================================
  // USER SETTINGS METHODS
  // ==========================================
  async getUserSettings(userId) {
    const store = this._readStore();
    if (!store.settings) store.settings = {};
    return store.settings[userId] || {
      user_id: userId,
      response_language: 'English',
      theme: 'dark',
      study_goal_hours: 4,
    };
  }

  async updateUserSettings(userId, newSettings) {
    const store = this._readStore();
    if (!store.settings) store.settings = {};
    store.settings[userId] = {
      user_id: userId,
      ...(store.settings[userId] || {}),
      ...newSettings,
      updated_at: new Date().toISOString(),
    };
    this._writeStore(store);
    return store.settings[userId];
  }
  async getDashboardStats(userId = 'default_user') {
    const store = this._readStore();
    const totalNotes = store.notes ? store.notes.length : 4;
    return {
      activeNotebooks: Math.max(new Set(store.notes.map((n) => n.subject)).size, 4),
      uploadedSources: 14,
      audioSynthesis: 3,
      solvedPYQs: 28,
      totalNotes,
    };
  }

  async getDashboardNotebooks(userId = 'default_user') {
    const store = this._readStore();
    const defaultNotebooks = [
      {
        id: 'nb-1',
        code: 'CS302',
        title: 'Operating Systems — Process Scheduling & Deadlocks',
        subject: 'Operating Systems',
        sourcesCount: 4,
        notesCount: 12,
        lastUpdated: '2 hours ago',
      },
      {
        id: 'nb-2',
        code: 'CS201',
        title: 'Data Structures — AVL Tree Rotations & Red-Black Trees',
        subject: 'Data Structures',
        sourcesCount: 3,
        notesCount: 8,
        lastUpdated: 'Yesterday',
      },
      {
        id: 'nb-3',
        code: 'EC401',
        title: 'Digital Signal Processing — FFT Decimation Algorithm',
        subject: 'Digital Signal Processing',
        sourcesCount: 2,
        notesCount: 5,
        lastUpdated: '3 days ago',
      },
      {
        id: 'nb-4',
        code: 'CS504',
        title: 'Database Systems — 3NF & BCNF Normalization PYQs',
        subject: 'Database Systems',
        sourcesCount: 5,
        notesCount: 15,
        lastUpdated: '4 days ago',
      },
    ];

    // If user has created custom notes, append notebooks dynamically
    const customNotes = (store.notes || []).filter((n) => !n.note_id.startsWith('note-'));
    customNotes.forEach((n, idx) => {
      defaultNotebooks.unshift({
        id: `nb-custom-${idx}`,
        code: n.code || 'GEN101',
        title: n.title || n.subject,
        subject: n.subject || 'General Study',
        sourcesCount: 1,
        notesCount: 1,
        lastUpdated: 'Just now',
      });
    });

    return defaultNotebooks;
  }
}

module.exports = new StorageService();
