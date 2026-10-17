const Note = require('../models/Note');
const { successResponse, errorResponse } = require('../utils/apiResponse');

// Default initial starter notes matching frontend requirements
const DEFAULT_INITIAL_NOTES = [
  {
    note_id: 'note-1',
    user_id: 'seed_user_1',
    code: 'CS302',
    subject: 'Operating Systems',
    chapter: 'Module 1: Process Scheduling & Deadlocks',
    title: 'Operating Systems — Process Scheduling & Deadlocks',
    college: 'IIT Bombay',
    branch: 'Computer Science',
    semester: 'Sem 5',
    note_type: 'Lecture Notes',
    description: "Complete breakdown of CPU scheduling algorithms (FCFS, SJF, Round Robin), Banker's algorithm for deadlock avoidance with resource allocation graph examples.",
    author: 'Rohan Sharma (IIT Bombay)',
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
  },
  {
    note_id: 'note-3',
    user_id: 'seed_user_3',
    code: 'EC401',
    subject: 'Digital Signal Processing',
    chapter: 'Module 4: FFT Decimation Algorithm',
    title: 'Digital Signal Processing — Fast Fourier Transform (FFT) Decimation',
    college: 'NIT Trichy',
    branch: 'Electronics',
    semester: 'Sem 6',
    note_type: 'PYQ & Solutions',
    description: 'Cooley-Tukey decimation-in-time (DIT) FFT algorithm derivation, butterfly flow diagrams, twiddle factor computation, and 5 years solved PYQs.',
    author: 'Ankit Patel (NIT Trichy)',
    likes_count: 98,
    read_time: '15 min read',
    formatted_content: '# Digital Signal Processing — Fast Fourier Transform\n\n## Overview\nStep-by-step reduction of DFT N^2 complexity to N log2 N using Radix-2 DIT-FFT decomposition.',
    summary: 'Radix-2 DIT FFT decomposition and butterfly diagrams.',
    keywords: ['FFT', 'DFT', 'Butterfly Diagram', 'Twiddle Factor', 'DSP'],
    duration: 50.0,
    full_content: {
      overview: 'Step-by-step reduction of DFT N^2 complexity to N log2 N using Radix-2 DIT-FFT decomposition.',
      key_topics: [
        'DFT Direct Computation vs Radix-2 DIT-FFT',
        'Bit-reversal indexing algorithm',
        '8-Point Butterfly Diagram computation',
        'Twiddle factor W_N^k symmetry and periodicity properties',
      ],
      pyqs: [
        {
          question: 'Compute 8-point DFT of sequence x(n) = {1, 2, 1, 0, 0, 0, 0, 0} using Radix-2 DIT-FFT algorithm.',
          year: 'Endsem 2025',
          solution: 'Bit reverse input array to {x(0), x(4), x(2), x(6), x(1), x(5), x(3), x(7)}. Perform 3 stages of butterfly computation.',
        },
      ],
    },
    is_public: true,
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
    },
    is_public: true,
  },
];

/**
 * @desc    Get all public notes for LandingPage with search, multi-faceted filtering & sorting
 * @route   GET /api/notes/public
 * @access  Public
 */
const getPublicNotes = async (req, res, next) => {
  try {
    const {
      college,
      branch,
      semester,
      subject,
      note_type,
      search,
      sort_by = 'Most Recent',
    } = req.query;

    // Check if collection has documents; seed defaults if empty
    const count = await Note.countDocuments();
    if (count === 0) {
      await Note.insertMany(DEFAULT_INITIAL_NOTES);
    }

    const query = { is_public: true };

    if (college && college !== 'All Colleges') {
      query.college = college;
    }
    if (branch && branch !== 'All Branches') {
      query.branch = branch;
    }
    if (semester && semester !== 'All Semesters') {
      query.semester = semester;
    }
    if (subject && subject !== 'All Subjects') {
      query.subject = subject;
    }
    if (note_type && note_type !== 'All Types') {
      query.note_type = note_type;
    }

    if (search && search.trim() !== '') {
      const searchRegex = new RegExp(search.trim(), 'i');
      query.$or = [
        { title: searchRegex },
        { subject: searchRegex },
        { chapter: searchRegex },
        { description: searchRegex },
        { author: searchRegex },
        { code: searchRegex },
      ];
    }

    let sortOptions = { created_at: -1 };
    if (sort_by === 'Most Liked') {
      sortOptions = { likes_count: -1, created_at: -1 };
    } else if (sort_by === 'Title A-Z') {
      sortOptions = { title: 1 };
    }

    const notes = await Note.find(query).sort(sortOptions);

    return successResponse(res, 200, 'Public study notes fetched successfully', {
      count: notes.length,
      notes,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get notes created by the authenticated user
 * @route   GET /api/notes
 * @access  Private
 */
const getUserNotes = async (req, res, next) => {
  try {
    const userId = req.user.user_id;
    const { subject } = req.query;

    const query = { user_id: userId };
    if (subject) {
      query.subject = subject;
    }

    const notes = await Note.find(query).sort({ created_at: -1 });

    return successResponse(res, 200, 'User notes fetched successfully', {
      count: notes.length,
      notes,
    });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Get single note by note_id
 * @route   GET /api/notes/:id
 * @access  Public / Optional Auth
 */
const getNoteById = async (req, res, next) => {
  try {
    const noteId = req.params.id;
    const note = await Note.findOne({
      $or: [{ note_id: noteId }, { _id: noteId.match(/^[0-9a-fA-F]{24}$/) ? noteId : null }],
    });

    if (!note) {
      return errorResponse(res, 404, 'Lecture note not found');
    }

    return successResponse(res, 200, 'Note retrieved successfully', { note });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Create a new lecture note
 * @route   POST /api/notes
 * @access  Private
 */
const createNote = async (req, res, next) => {
  try {
    const userId = req.user ? req.user.user_id : (req.body.user_id || 'default_user');
    const {
      subject,
      chapter,
      formatted_content,
      title,
      description,
      summary,
      keywords,
      duration,
      audio_file_id,
      code,
      college,
      branch,
      semester,
      note_type,
      full_content,
      is_public,
    } = req.body;

    if (!subject || !chapter || !formatted_content) {
      return errorResponse(res, 400, 'Please provide subject, chapter, and formatted_content');
    }

    const note = await Note.create({
      user_id: userId,
      subject,
      chapter,
      title: title || `${subject} — ${chapter}`,
      description: description || summary || '',
      formatted_content,
      summary: summary || '',
      keywords: keywords || [],
      duration: duration || 0.0,
      audio_file_id: audio_file_id || null,
      code: code || 'CS101',
      college: college || (req.user ? req.user.college : 'IIT Bombay'),
      branch: branch || (req.user ? req.user.branch : 'Computer Science'),
      semester: semester || (req.user ? req.user.semester : 'Sem 5'),
      note_type: note_type || 'Lecture Notes',
      author: req.user ? `${req.user.name} (${req.user.college})` : 'StudySnap Student',
      full_content: full_content || {
        overview: summary || '',
        key_topics: keywords || [],
        pyqs: [],
        formulas: [],
      },
      is_public: is_public !== undefined ? is_public : true,
    });

    return successResponse(res, 201, 'Lecture note created successfully', { note });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Update an existing lecture note
 * @route   PUT /api/notes/:id
 * @access  Private
 */
const updateNote = async (req, res, next) => {
  try {
    const noteId = req.params.id;
    const updateData = { ...req.body, updated_at: new Date() };

    const note = await Note.findOneAndUpdate(
      { note_id: noteId },
      { $set: updateData },
      { new: true, runValidators: true }
    );

    if (!note) {
      return errorResponse(res, 404, 'Note not found');
    }

    return successResponse(res, 200, 'Note updated successfully', { note });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Delete a lecture note
 * @route   DELETE /api/notes/:id
 * @access  Private
 */
const deleteNote = async (req, res, next) => {
  try {
    const noteId = req.params.id;
    const note = await Note.findOneAndDelete({ note_id: noteId });

    if (!note) {
      return errorResponse(res, 404, 'Note not found');
    }

    return successResponse(res, 200, 'Note deleted successfully', { note_id: noteId });
  } catch (error) {
    next(error);
  }
};

/**
 * @desc    Toggle like on a lecture note
 * @route   POST /api/notes/:id/like
 * @access  Private
 */
const toggleLikeNote = async (req, res, next) => {
  try {
    const noteId = req.params.id;
    const userId = req.user ? req.user.user_id : 'anonymous_user';

    const note = await Note.findOne({ note_id: noteId });
    if (!note) {
      return errorResponse(res, 404, 'Note not found');
    }

    const hasLiked = note.liked_by && note.liked_by.includes(userId);

    if (hasLiked) {
      note.liked_by = note.liked_by.filter((id) => id !== userId);
      note.likes_count = Math.max(0, (note.likes_count || 1) - 1);
    } else {
      if (!note.liked_by) note.liked_by = [];
      note.liked_by.push(userId);
      note.likes_count = (note.likes_count || 0) + 1;
    }

    await note.save();

    return successResponse(res, 200, hasLiked ? 'Note unliked' : 'Note liked', {
      likes_count: note.likes_count,
      is_liked: !hasLiked,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = {
  getPublicNotes,
  getUserNotes,
  getNoteById,
  createNote,
  updateNote,
  deleteNote,
  toggleLikeNote,
};
