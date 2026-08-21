export interface PublicNote {
  id: string;
  code: string; // e.g. CS302
  title: string;
  date: string;
  college: string;
  branch: string;
  semester: string;
  subject: string;
  noteType: 'Lecture Notes' | 'PYQ & Solutions' | 'Formula Sheet' | 'Lab Manual';
  description: string;
  author: string;
  likesCount: number;
  readTime: string;
  fullContent: {
    overview: string;
    keyTopics: string[];
    pyqs: { question: string; year: string; solution: string }[];
    formulas?: string[];
  };
}

export const MOCK_PUBLIC_NOTES: PublicNote[] = [
  {
    id: 'note-1',
    code: 'CS302',
    date: '2026-07-28',
    college: 'IIT Bombay',
    branch: 'Computer Science',
    semester: 'Sem 5',
    subject: 'Operating Systems',
    noteType: 'Lecture Notes',
    title: 'Operating Systems — Process Scheduling & Deadlocks',
    description: "Complete breakdown of CPU scheduling algorithms (FCFS, SJF, Round Robin), Banker's algorithm for deadlock avoidance with resource allocation graph examples.",
    author: 'Rohan Sharma (IIT Bombay)',
    likesCount: 142,
    readTime: '8 min read',
    fullContent: {
      overview: 'This comprehensive guide covers kernel process management, CPU scheduler implementations in Linux, deadlock conditions (Coffman conditions), and Banker algorithm matrices.',
      keyTopics: [
        'FCFS, SJF, SRTF, and Round Robin Gantt Chart examples',
        'Multi-level Queue Scheduling & Feedback Queues',
        '4 Necessary Conditions for Deadlock: Mutual Exclusion, Hold & Wait, No Preemption, Circular Wait',
        "Resource Allocation Graph (RAG) reduction and Banker's Safety Algorithm"
      ],
      pyqs: [
        {
          question: 'Consider 5 processes P0 through P4 with Allocation and Max matrices. Calculate the Need matrix and verify if the system is in a safe state.',
          year: 'Endsem 2025',
          solution: 'Need = Max - Allocation. System is in safe state with sequence <P1, P3, P4, P0, P2>.'
        },
        {
          question: 'Differentiate between preemptive and non-preemptive scheduling with time-quantum trade-offs.',
          year: 'Midsem 2024',
          solution: 'Preemptive scheduling interrupts running tasks when higher priority arrives. Quantum too small increases context switch overhead; quantum too large degrades to FCFS.'
        }
      ],
      formulas: [
        'Turnaround Time (TAT) = Completion Time - Arrival Time',
        'Waiting Time (WT) = Turnaround Time - Burst Time',
        'Need[i][j] = Max[i][j] - Allocation[i][j]'
      ]
    }
  },
  {
    id: 'note-2',
    code: 'CS201',
    date: '2026-07-27',
    college: 'PES University',
    branch: 'Computer Science',
    semester: 'Sem 3',
    subject: 'Data Structures',
    noteType: 'Formula Sheet',
    title: 'Data Structures — AVL Tree Rotations & Red-Black Trees',
    description: 'Complete step-by-step balance factor derivations for Single (LL, RR) and Double (LR, RL) rotations with time complexity proofs and code templates.',
    author: 'Priya Verma (PES University)',
    likesCount: 215,
    readTime: '12 min read',
    fullContent: {
      overview: 'Detailed visual cheat-sheet and code notes for self-balancing binary search trees, balance factor computation, and Red-Black tree insertion re-coloring rules.',
      keyTopics: [
        'Balance Factor = Height(Left Subtree) - Height(Right Subtree) ∈ {-1, 0, +1}',
        'Single Rotations: LL Rotation (Right Rotate), RR Rotation (Left Rotate)',
        'Double Rotations: LR Rotation (Left-Right), RL Rotation (Right-Left)',
        'Red-Black Tree Properties: Root is black, No adjacent red nodes, Black-height equality'
      ],
      pyqs: [
        {
          question: 'Insert keys 14, 17, 11, 7, 53, 4, 13 into an empty AVL tree and show rebalancing at each step.',
          year: 'Endsem 2024',
          solution: 'Inserting 4 triggers LL imbalance at node 11. Right rotate at 11 yields balanced tree with root 11.'
        }
      ],
      formulas: [
        'Height of AVL tree h ≤ 1.44 log2(N + 2)',
        'Search / Insert / Delete Time Complexity = O(log N)'
      ]
    }
  },
  {
    id: 'note-3',
    code: 'EC401',
    date: '2026-07-25',
    college: 'NIT Trichy',
    branch: 'Electronics',
    semester: 'Sem 6',
    subject: 'Digital Signal Processing',
    noteType: 'PYQ & Solutions',
    title: 'Digital Signal Processing — Fast Fourier Transform (FFT) Decimation',
    description: 'Cooley-Tukey decimation-in-time (DIT) FFT algorithm derivation, butterfly flow diagrams, twiddle factor computation, and 5 years solved PYQs.',
    author: 'Ankit Patel (NIT Trichy)',
    likesCount: 98,
    readTime: '15 min read',
    fullContent: {
      overview: 'Step-by-step reduction of DFT N^2 complexity to N log2 N using Radix-2 DIT-FFT decomposition.',
      keyTopics: [
        'DFT Direct Computation vs Radix-2 DIT-FFT',
        'Bit-reversal indexing algorithm',
        '8-Point Butterfly Diagram computation',
        'Twiddle factor W_N^k symmetry and periodicity properties'
      ],
      pyqs: [
        {
          question: 'Compute 8-point DFT of sequence x(n) = {1, 2, 1, 0, 0, 0, 0, 0} using Radix-2 DIT-FFT algorithm.',
          year: 'Endsem 2025',
          solution: 'Bit reverse input array to {x(0), x(4), x(2), x(6), x(1), x(5), x(3), x(7)}. Perform 3 stages of butterfly computation.'
        }
      ]
    }
  },
  {
    id: 'note-4',
    code: 'CS504',
    date: '2026-07-22',
    college: 'BITS Pilani',
    branch: 'Computer Science',
    semester: 'Sem 5',
    subject: 'Database Systems',
    noteType: 'PYQ & Solutions',
    title: 'Database Management — 3NF & BCNF Normalization Solved PYQs',
    description: 'Step-by-step resolution of functional dependencies, minimal cover finding, 3NF and BCNF decomposition with lossless join and dependency preservation proofs.',
    author: 'Sneha Reddy (BITS Pilani)',
    likesCount: 180,
    readTime: '10 min read',
    fullContent: {
      overview: 'Master relational schema normalization with practical algorithms to test 2NF, 3NF, BCNF, and dependency preserving decompositions.',
      keyTopics: [
        'Candidate Key determination using Attribute Closure',
        'Minimal Cover / Canonical Cover calculation',
        '3NF Synthesis Algorithm',
        'BCNF Decomposition Algorithm (Checking X -> Y where X is superkey)'
      ],
      pyqs: [
        {
          question: 'Given R(A,B,C,D,E) with F = {A->BC, CD->E, B->D, E->A}. Find candidate keys and decompose R into BCNF.',
          year: 'Endsem 2024',
          solution: 'Candidate keys are {A}, {E}, {BC}. Relation is in 3NF but not BCNF due to B->D. Decomposed into R1(B,D) and R2(A,B,C,E).'
        }
      ]
    }
  },
  {
    id: 'note-5',
    code: 'AI602',
    date: '2026-07-20',
    college: 'DTU',
    branch: 'Information Technology',
    semester: 'Sem 7',
    subject: 'Machine Learning',
    noteType: 'Lecture Notes',
    title: 'Machine Learning — Backpropagation & Gradient Descent Derivations',
    description: 'Comprehensive matrix calculus for neural net backpropagation, loss functions (MSE, Cross-Entropy), and Adam vs SGD optimizer comparison.',
    author: 'Vikram Malhotra (DTU)',
    likesCount: 310,
    readTime: '14 min read',
    fullContent: {
      overview: 'Mathematical breakdown of forward pass, activation functions (ReLU, Sigmoid, Softmax), chain rule partial derivatives, and weight update equations.',
      keyTopics: [
        'Multi-Layer Perceptron (MLP) Architecture',
        'Chain Rule for Backpropagation: dL/dw = dL/da * da/dz * dz/dw',
        'Vanishing and Exploding Gradient solutions (He / Xavier Initialization)',
        'Optimization: SGD, Momentum, RMSProp, Adam Optimizer'
      ],
      pyqs: [
        {
          question: 'Derive the weight update rule for a 2-layer neural network using Cross-Entropy loss and Sigmoid output activation.',
          year: 'Midsem 2025',
          solution: 'dL/dz_out = (a_out - y). The derivative simplifies gracefully without division by zero.'
        }
      ]
    }
  }
];
