// =============================================================================
// StudySnap AI — Academic Data Models & Validation Utilities
// =============================================================================

export interface DegreeOption {
  label: string;
  value: string;
  standardDurationYears: number;
}

export const DEGREE_OPTIONS: DegreeOption[] = [
  { label: 'B.Tech / B.E. (Bachelor of Technology / Engineering)', value: 'B.Tech / B.E.', standardDurationYears: 4 },
  { label: 'B.Sc. (Bachelor of Science)', value: 'B.Sc.', standardDurationYears: 3 },
  { label: 'BCA (Bachelor of Computer Applications)', value: 'BCA', standardDurationYears: 3 },
  { label: 'BBA (Bachelor of Business Administration)', value: 'BBA', standardDurationYears: 3 },
  { label: 'B.Com. (Bachelor of Commerce)', value: 'B.Com.', standardDurationYears: 3 },
  { label: 'M.Tech / M.E. (Master of Technology / Engineering)', value: 'M.Tech / M.E.', standardDurationYears: 2 },
  { label: 'M.Sc. (Master of Science)', value: 'M.Sc.', standardDurationYears: 2 },
  { label: 'MCA (Master of Computer Applications)', value: 'MCA', standardDurationYears: 2 },
  { label: 'Diploma in Engineering / Technology', value: 'Diploma', standardDurationYears: 3 },
  { label: 'Other Degree / Program', value: 'Other', standardDurationYears: 4 },
];

export const BRANCH_OPTIONS = [
  'Computer Science Engineering (CSE)',
  'Artificial Intelligence and Machine Learning (AIML)',
  'Artificial Intelligence and Data Science (AI&DS)',
  'Information Technology (IT)',
  'Electronics and Telecommunication (ENTC)',
  'Electronics Engineering',
  'Electrical Engineering',
  'Mechanical Engineering',
  'Civil Engineering',
  'Chemical Engineering',
  'Data Science & Analytics',
  'Software Development',
  'Cybersecurity & Network Defense',
  'Business Administration & Management',
  'Accounting & Financial Studies',
  'Computer Applications & Informatics',
  'Mathematics & Computing',
  'Physical Sciences & Physics',
  'Chemical Sciences & Chemistry',
  'Biotechnology & Life Sciences',
  'Other Branch / Department',
];

export const ACADEMIC_YEARS = [
  'First Year',
  'Second Year',
  'Third Year',
  'Fourth Year',
  'Other',
];

/**
 * Get available semester numbers based on selected academic year
 */
export const getAvailableSemesters = (academicYear: string): string[] => {
  switch (academicYear) {
    case 'First Year':
      return ['Semester 1', 'Semester 2'];
    case 'Second Year':
      return ['Semester 3', 'Semester 4'];
    case 'Third Year':
      return ['Semester 5', 'Semester 6'];
    case 'Fourth Year':
      return ['Semester 7', 'Semester 8'];
    case 'Other':
    default:
      return [
        'Semester 1', 'Semester 2', 'Semester 3', 'Semester 4',
        'Semester 5', 'Semester 6', 'Semester 7', 'Semester 8',
        'Semester 9', 'Semester 10'
      ];
  }
};

/**
 * Generate a realistic list of expected graduation years (current - 1 to current + 6)
 */
export const getGraduationYears = (): string[] => {
  const currentYear = new Date().getFullYear();
  const years: string[] = [];
  for (let y = currentYear - 1; y <= currentYear + 6; y++) {
    years.push(y.toString());
  }
  return years;
};

/**
 * Validate academic email or personal email format
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email.trim());
};

/**
 * Password validation rule: min 6 chars
 */
export const validatePassword = (password: string): { valid: boolean; message?: string } => {
  if (!password) {
    return { valid: false, message: 'Password is required' };
  }
  if (password.length < 6) {
    return { valid: false, message: 'Password must be at least 6 characters' };
  }
  return { valid: true };
};

/**
 * Format student display badge: e.g. "B.Tech • Computer Science Engineering"
 */
export const formatStudentProgramBadge = (
  degree?: string | null,
  customDegree?: string | null,
  branch?: string | null,
  customBranch?: string | null
): string => {
  const d = degree === 'Other' ? (customDegree || 'Custom Degree') : (degree || 'Student');
  const b = branch === 'Other Branch / Department' || branch === 'Other'
    ? (customBranch || 'Specialization')
    : (branch ? branch.replace(/\s*\(.*?\)\s*/g, '') : '');
  
  if (b) {
    return `${d} • ${b}`;
  }
  return d;
};
