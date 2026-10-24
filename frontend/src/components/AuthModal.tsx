import React, { useState, useEffect } from 'react';
import {
  X, Sparkles, Loader2, AlertCircle, CheckCircle2,
  ArrowRight, ArrowLeft, GraduationCap, User, Mail,
  Lock, Building2, BookOpen, Calendar, CheckSquare, Eye, EyeOff
} from 'lucide-react';
import { API_GATEWAY_URL, setAuthToken } from '../config/api';
import {
  isSupabaseConfigured,
  signInWithEmail,
  signUpStudent,
  resetPasswordForEmail,
  fetchStudentProfile
} from '../config/supabase';
import type { StudentSignupData, StudentProfile } from '../config/supabase';
import {
  DEGREE_OPTIONS,
  BRANCH_OPTIONS,
  ACADEMIC_YEARS,
  getAvailableSemesters,
  getGraduationYears,
  isValidEmail,
  validatePassword
} from '../data/academicData';
import './AuthModal.css';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onLoginSuccess: (user: any) => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({
  isOpen,
  onClose,
  onLoginSuccess
}) => {
  // Modal mode: 'login' | 'signup'
  const [isRegister, setIsRegister] = useState(false);
  // Signup step: 1 (Account Credentials) | 2 (Academic Profile)
  const [signupStep, setSignupStep] = useState<1 | 2>(1);

  // Form Fields
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Academic Details
  const [college, setCollege] = useState('DKTE Society\'s Textile and Engineering Institute');
  const [degreeProgram, setDegreeProgram] = useState('B.Tech / B.E.');
  const [customDegree, setCustomDegree] = useState('');
  const [branch, setBranch] = useState('Computer Science Engineering (CSE)');
  const [customBranch, setCustomBranch] = useState('');
  const [academicYear, setAcademicYear] = useState('Third Year');
  const [semester, setSemester] = useState('Semester 5');
  const [graduationYear, setGraduationYear] = useState('2026');
  const [rollNumber, setRollNumber] = useState('');
  const [agreeTerms, setAgreeTerms] = useState(false);

  // Feedback & State
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);

  // Dynamic available semesters when academic year changes
  const availableSemesters = getAvailableSemesters(academicYear);
  const graduationYears = getGraduationYears();

  // Reset semester if selected semester is not valid for new year
  useEffect(() => {
    if (!availableSemesters.includes(semester)) {
      setSemester(availableSemesters[0] || 'Semester 1');
    }
  }, [academicYear, availableSemesters, semester]);

  if (!isOpen) return null;

  // Step 1 Validation
  const validateStep1 = (): boolean => {
    setErrorMsg(null);
    if (!name.trim()) {
      setErrorMsg('Please enter your full legal name.');
      return false;
    }
    if (!email.trim() || !isValidEmail(email)) {
      setErrorMsg('Please enter a valid email address.');
      return false;
    }
    const passCheck = validatePassword(password);
    if (!passCheck.valid) {
      setErrorMsg(passCheck.message || 'Password must be at least 6 characters.');
      return false;
    }
    if (password !== confirmPassword) {
      setErrorMsg('Passwords do not match. Please re-enter.');
      return false;
    }
    return true;
  };

  // Step 2 Validation
  const validateStep2 = (): boolean => {
    setErrorMsg(null);
    if (!college.trim()) {
      setErrorMsg('Please enter your college or university name.');
      return false;
    }
    if (degreeProgram === 'Other' && !customDegree.trim()) {
      setErrorMsg('Please specify your degree or program name.');
      return false;
    }
    if ((branch === 'Other Branch / Department' || branch === 'Other') && !customBranch.trim()) {
      setErrorMsg('Please specify your department or branch of study.');
      return false;
    }
    if (!agreeTerms) {
      setErrorMsg('You must agree to the Terms of Service and Privacy Policy to create an account.');
      return false;
    }
    return true;
  };

  const handleNextStep = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateStep1()) {
      setSignupStep(2);
      setErrorMsg(null);
    }
  };

  const handlePreviousStep = () => {
    setErrorMsg(null);
    setSignupStep(1);
  };

  const handleRegisterSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateStep2()) return;

    setLoading(true);
    setErrorMsg(null);
    setInfoMsg(null);

    const signupPayload: StudentSignupData = {
      fullName: name.trim(),
      email: email.trim(),
      password,
      collegeName: college.trim(),
      degreeProgram,
      customDegree: degreeProgram === 'Other' ? customDegree.trim() : undefined,
      branch,
      customBranch: (branch === 'Other Branch / Department' || branch === 'Other') ? customBranch.trim() : undefined,
      academicYear,
      semester,
      graduationYear,
      rollNumber: rollNumber.trim() || undefined,
    };

    if (isSupabaseConfigured()) {
      try {
        const authRes = await signUpStudent(signupPayload);

        if (authRes.session && authRes.user) {
          setAuthToken(authRes.session.access_token);
          const studentProfile: StudentProfile = {
            id: authRes.user.id,
            full_name: name.trim(),
            display_name: name.trim(),
            email: email.trim(),
            college_name: college.trim(),
            college: college.trim(),
            degree_program: degreeProgram,
            custom_degree: degreeProgram === 'Other' ? customDegree.trim() : null,
            branch: branch,
            custom_branch: (branch === 'Other Branch / Department' || branch === 'Other') ? customBranch.trim() : null,
            academic_year: academicYear,
            semester: semester,
            graduation_year: graduationYear,
            roll_number: rollNumber.trim() || null,
            email_verified: Boolean(authRes.user.email_confirmed_at),
          };
          localStorage.setItem('studysnap_user', JSON.stringify(studentProfile));
          onLoginSuccess(studentProfile);
          onClose();
        } else {
          // Verification email required
          setIsRegister(false);
          setSignupStep(1);
          setInfoMsg(`Registration complete! A verification link has been sent to ${email}. Please check your inbox and verify your email to log in.`);
        }
      } catch (err: any) {
        console.error('[StudySnap Auth] Signup error:', err);
        const rawMsg = err.message || '';
        if (rawMsg.toLowerCase().includes('rate limit') || err.status === 429) {
          setErrorMsg('Supabase email rate limit reached (free tier limit of 3 emails/hr). In your Supabase Dashboard, go to Authentication > Providers > Email and turn OFF "Confirm email" for instant signups.');
        } else if (rawMsg.toLowerCase().includes('user already registered') || rawMsg.toLowerCase().includes('already exists')) {
          setErrorMsg('An account with this email address already exists. Please log in instead.');
        } else {
          setErrorMsg(rawMsg || 'Failed to create student account. Please verify your details.');
        }
      } finally {
        setLoading(false);
      }
      return;
    }

    // Local Gateway Fallback
    try {
      const res = await fetch(`${API_GATEWAY_URL}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(signupPayload)
      });
      const data = await res.json();
      if (res.ok && data.success && data.data?.token) {
        setAuthToken(data.data.token);
        const userObj = {
          name: name.trim(),
          full_name: name.trim(),
          email: email.trim(),
          college: college.trim(),
          college_name: college.trim(),
          degree_program: degreeProgram,
          branch,
          academic_year: academicYear,
          semester,
          graduation_year: graduationYear,
        };
        localStorage.setItem('studysnap_user', JSON.stringify(userObj));
        onLoginSuccess(userObj);
        onClose();
      } else {
        setErrorMsg(data.message || 'Registration failed. Please check credentials.');
      }
    } catch {
      // Offline fallback
      const offlineUser = {
        name: name.trim() || 'Student User',
        full_name: name.trim() || 'Student User',
        email: email.trim(),
        college: college.trim(),
        college_name: college.trim(),
        degree_program: degreeProgram,
        branch,
        academic_year: academicYear,
        semester,
        graduation_year: graduationYear,
      };
      localStorage.setItem('studysnap_user', JSON.stringify(offlineUser));
      onLoginSuccess(offlineUser);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setErrorMsg('Please enter both your email address and password.');
      return;
    }

    setLoading(true);
    setErrorMsg(null);
    setInfoMsg(null);

    if (isSupabaseConfigured()) {
      try {
        const { data, error } = await signInWithEmail(email, password);
        if (error) throw error;

        if (data.session && data.user) {
          setAuthToken(data.session.access_token);
          // Fetch student profile record from database
          const profile = await fetchStudentProfile(data.user.id);
          const userObj = profile || {
            id: data.user.id,
            name: data.user.user_metadata?.full_name || data.user.user_metadata?.name || email.split('@')[0],
            full_name: data.user.user_metadata?.full_name || data.user.user_metadata?.name || email.split('@')[0],
            email: data.user.email || email,
            college: data.user.user_metadata?.college_name || data.user.user_metadata?.college || 'DKTE',
            college_name: data.user.user_metadata?.college_name || data.user.user_metadata?.college || 'DKTE',
            degree_program: data.user.user_metadata?.degree_program || 'B.Tech / B.E.',
            branch: data.user.user_metadata?.branch || 'Computer Science Engineering (CSE)',
            academic_year: data.user.user_metadata?.academic_year || 'Third Year',
            semester: data.user.user_metadata?.semester || 'Semester 5',
            graduation_year: data.user.user_metadata?.graduation_year || '2026',
            email_verified: Boolean(data.user.email_confirmed_at),
          };
          localStorage.setItem('studysnap_user', JSON.stringify(userObj));
          onLoginSuccess(userObj);
          onClose();
        }
      } catch (err: any) {
        console.error('[StudySnap Auth] Login error:', err);
        const rawMsg = err.message || '';
        if (rawMsg.toLowerCase().includes('invalid login credentials')) {
          setErrorMsg('Invalid email or password. Please verify your credentials.');
        } else if (rawMsg.toLowerCase().includes('email not confirmed')) {
          setErrorMsg('Your email has not been verified yet. Please check your inbox for the confirmation link.');
        } else {
          setErrorMsg(rawMsg || 'Login failed. Please check your credentials.');
        }
      } finally {
        setLoading(false);
      }
      return;
    }

    // Local Gateway fallback
    try {
      const res = await fetch(`${API_GATEWAY_URL}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok && data.success && data.data?.token) {
        setAuthToken(data.data.token);
        const userObj = data.data.user || {
          name: email.split('@')[0],
          email,
          college: 'DKTE'
        };
        localStorage.setItem('studysnap_user', JSON.stringify(userObj));
        onLoginSuccess(userObj);
        onClose();
      } else {
        setErrorMsg(data.message || 'Invalid email or password.');
      }
    } catch {
      const fallbackUser = {
        name: email.split('@')[0] || 'Student',
        email,
        college: 'DKTE'
      };
      localStorage.setItem('studysnap_user', JSON.stringify(fallbackUser));
      onLoginSuccess(fallbackUser);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemoLogin = async () => {
    setLoading(true);
    setErrorMsg(null);
    setInfoMsg(null);
    try {
      const res = await fetch(`${API_GATEWAY_URL}/api/auth/demo-login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await res.json();
      if (res.ok && data.success && data.data?.token) {
        setAuthToken(data.data.token);
        const userObj = data.data.user || {
          id: 'demo-student-id',
          name: 'Rohan Sharma',
          full_name: 'Rohan Sharma',
          email: 'rohan.sharma@iitb.ac.in',
          college: 'DKTE Society\'s Textile and Engineering Institute',
          college_name: 'DKTE Society\'s Textile and Engineering Institute',
          degree_program: 'B.Tech / B.E.',
          branch: 'Computer Science Engineering (CSE)',
          academic_year: 'Third Year',
          semester: 'Semester 5',
          graduation_year: '2026',
          roll_number: '22CS1049',
          email_verified: true,
        };
        localStorage.setItem('studysnap_user', JSON.stringify(userObj));
        onLoginSuccess(userObj);
        onClose();
      } else {
        throw new Error(data.message || 'Demo login failed');
      }
    } catch {
      const demoUser = {
        id: 'demo-student-id',
        name: 'Rohan Sharma',
        full_name: 'Rohan Sharma',
        email: 'rohan.sharma@iitb.ac.in',
        college: 'DKTE Society\'s Textile and Engineering Institute',
        college_name: 'DKTE Society\'s Textile and Engineering Institute',
        degree_program: 'B.Tech / B.E.',
        branch: 'Computer Science Engineering (CSE)',
        academic_year: 'Third Year',
        semester: 'Semester 5',
        graduation_year: '2026',
        roll_number: '22CS1049',
        email_verified: true,
      };
      localStorage.setItem('studysnap_user', JSON.stringify(demoUser));
      onLoginSuccess(demoUser);
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="auth-modal-title">
      <div className={`auth-modal ${isRegister ? 'auth-modal-signup' : ''}`} onClick={e => e.stopPropagation()}>
        <button className="auth-close" onClick={onClose} disabled={loading} aria-label="Close modal">
          <X size={20} />
        </button>

        {/* Modal Header */}
        <div className="auth-header">
          <div className="auth-badge-icon">
            <GraduationCap size={22} />
          </div>
          <h2 id="auth-modal-title" className="auth-title">
            {isRegister
              ? (signupStep === 1 ? 'Create Student Account' : 'Academic Profile')
              : 'Log in to StudySnap AI'}
          </h2>
          <p className="auth-subtitle">
            {isRegister
              ? (signupStep === 1
                ? 'Join verified students turning lectures into exam-ready notes'
                : 'Personalize your AI study assistant with your university curriculum')
              : 'Turn your lecture audio and sources into exam-ready notes'}
          </p>
        </div>

        {/* Signup Stepper Indicator */}
        {isRegister && (
          <div className="auth-stepper">
            <div className={`auth-step-item ${signupStep >= 1 ? 'active' : ''} ${signupStep > 1 ? 'completed' : ''}`}>
              <div className="auth-step-dot">
                {signupStep > 1 ? <CheckCircle2 size={14} /> : '1'}
              </div>
              <span className="auth-step-label">Account</span>
            </div>
            <div className={`auth-step-line ${signupStep === 2 ? 'active' : ''}`} />
            <div className={`auth-step-item ${signupStep === 2 ? 'active' : ''}`}>
              <div className="auth-step-dot">2</div>
              <span className="auth-step-label">Academic Profile</span>
            </div>
          </div>
        )}

        {/* Alerts */}
        {errorMsg && (
          <div className="auth-alert auth-alert-error" role="alert">
            <AlertCircle size={16} className="auth-alert-icon" />
            <span>{errorMsg}</span>
          </div>
        )}

        {infoMsg && (
          <div className="auth-alert auth-alert-info" role="status">
            <Sparkles size={16} className="auth-alert-icon" />
            <span>{infoMsg}</span>
          </div>
        )}

        {/* ── LOGIN FORM ── */}
        {!isRegister && (
          <form className="auth-form" onSubmit={handleLoginSubmit}>
            <div className="auth-field">
              <label className="auth-label" htmlFor="login-email">
                <Mail size={14} /> Email Address
              </label>
              <input
                id="login-email"
                type="email"
                className="auth-input"
                placeholder="student@college.edu.in"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>

            <div className="auth-field">
              <div className="auth-field-row">
                <label className="auth-label" htmlFor="login-password">
                  <Lock size={14} /> Password
                </label>
                <button
                  type="button"
                  className="auth-forgot-btn"
                  onClick={async () => {
                    if (!email || !isValidEmail(email)) {
                      setErrorMsg('Please enter your email above to receive a password reset link.');
                      return;
                    }
                    try {
                      setLoading(true);
                      await resetPasswordForEmail(email);
                      setInfoMsg(`Password recovery instructions sent to ${email}. Please check your inbox.`);
                      setErrorMsg(null);
                    } catch (err: any) {
                      setErrorMsg(err.message || 'Failed to send password reset email.');
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  Forgot password?
                </button>
              </div>
              <div className="auth-password-wrapper">
                <input
                  id="login-password"
                  type={showPassword ? 'text' : 'password'}
                  className="auth-input"
                  placeholder="••••••••"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="auth-eye-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <button type="submit" className="auth-submit-btn" disabled={loading}>
              {loading ? (
                <span className="auth-btn-loader">
                  <Loader2 size={16} className="animate-spin" /> Logging In...
                </span>
              ) : (
                'Log In to Dashboard'
              )}
            </button>
          </form>
        )}

        {/* ── SIGNUP FORM STEP 1: ACCOUNT DETAILS ── */}
        {isRegister && signupStep === 1 && (
          <form className="auth-form" onSubmit={handleNextStep}>
            <div className="auth-field">
              <label className="auth-label" htmlFor="signup-name">
                <User size={14} /> Full Name <span className="auth-required">*</span>
              </label>
              <input
                id="signup-name"
                type="text"
                className="auth-input"
                placeholder="Rohan Sharma"
                value={name}
                onChange={e => setName(e.target.value)}
                required
                autoComplete="name"
              />
            </div>

            <div className="auth-field">
              <label className="auth-label" htmlFor="signup-email">
                <Mail size={14} /> Student / College Email <span className="auth-required">*</span>
              </label>
              <input
                id="signup-email"
                type="email"
                className="auth-input"
                placeholder="name@college.edu.in"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
              <span className="auth-field-hint">Used for login, syllabus personalization, and notebook backups.</span>
            </div>

            <div className="auth-field-grid">
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-password">
                  <Lock size={14} /> Password <span className="auth-required">*</span>
                </label>
                <div className="auth-password-wrapper">
                  <input
                    id="signup-password"
                    type={showPassword ? 'text' : 'password'}
                    className="auth-input"
                    placeholder="Min 6 characters"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    minLength={6}
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    className="auth-eye-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-confirm-password">
                  <Lock size={14} /> Confirm Password <span className="auth-required">*</span>
                </label>
                <div className="auth-password-wrapper">
                  <input
                    id="signup-confirm-password"
                    type={showConfirmPassword ? 'text' : 'password'}
                    className="auth-input"
                    placeholder="Re-enter password"
                    value={confirmPassword}
                    onChange={e => setConfirmPassword(e.target.value)}
                    required
                    minLength={6}
                    autoComplete="new-password"
                  />
                  <button
                    type="button"
                    className="auth-eye-toggle"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                  >
                    {showConfirmPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </div>
            </div>

            <button type="submit" className="auth-submit-btn">
              <span>Continue to Academic Details</span>
              <ArrowRight size={16} />
            </button>
          </form>
        )}

        {/* ── SIGNUP FORM STEP 2: ACADEMIC DETAILS ── */}
        {isRegister && signupStep === 2 && (
          <form className="auth-form" onSubmit={handleRegisterSubmit}>
            <div className="auth-field">
              <label className="auth-label" htmlFor="signup-college">
                <Building2 size={14} /> College / University Name <span className="auth-required">*</span>
              </label>
              <input
                id="signup-college"
                type="text"
                className="auth-input"
                placeholder="DKTE Society's Textile & Engineering Institute"
                value={college}
                onChange={e => setCollege(e.target.value)}
                required
              />
            </div>

            <div className="auth-field-grid">
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-degree">
                  <BookOpen size={14} /> Degree / Program <span className="auth-required">*</span>
                </label>
                <select
                  id="signup-degree"
                  className="auth-input auth-select"
                  value={degreeProgram}
                  onChange={e => setDegreeProgram(e.target.value)}
                  required
                >
                  {DEGREE_OPTIONS.map(d => (
                    <option key={d.value} value={d.value}>
                      {d.label}
                    </option>
                  ))}
                </select>
              </div>

              {degreeProgram === 'Other' && (
                <div className="auth-field">
                  <label className="auth-label" htmlFor="signup-custom-degree">
                    Specify Degree <span className="auth-required">*</span>
                  </label>
                  <input
                    id="signup-custom-degree"
                    type="text"
                    className="auth-input"
                    placeholder="e.g. B.Des / B.Arch"
                    value={customDegree}
                    onChange={e => setCustomDegree(e.target.value)}
                    required
                  />
                </div>
              )}

              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-branch">
                  <GraduationCap size={14} /> Branch / Department <span className="auth-required">*</span>
                </label>
                <select
                  id="signup-branch"
                  className="auth-input auth-select"
                  value={branch}
                  onChange={e => setBranch(e.target.value)}
                  required
                >
                  {BRANCH_OPTIONS.map(b => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
              </div>

              {(branch === 'Other Branch / Department' || branch === 'Other') && (
                <div className="auth-field">
                  <label className="auth-label" htmlFor="signup-custom-branch">
                    Specify Branch <span className="auth-required">*</span>
                  </label>
                  <input
                    id="signup-custom-branch"
                    type="text"
                    className="auth-input"
                    placeholder="e.g. Aerospace Engineering"
                    value={customBranch}
                    onChange={e => setCustomBranch(e.target.value)}
                    required
                  />
                </div>
              )}
            </div>

            <div className="auth-field-grid auth-field-triplet">
              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-year">
                  <Calendar size={14} /> Academic Year <span className="auth-required">*</span>
                </label>
                <select
                  id="signup-year"
                  className="auth-input auth-select"
                  value={academicYear}
                  onChange={e => setAcademicYear(e.target.value)}
                  required
                >
                  {ACADEMIC_YEARS.map(y => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-semester">
                  Semester <span className="auth-required">*</span>
                </label>
                <select
                  id="signup-semester"
                  className="auth-input auth-select"
                  value={semester}
                  onChange={e => setSemester(e.target.value)}
                  required
                >
                  {availableSemesters.map(s => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>

              <div className="auth-field">
                <label className="auth-label" htmlFor="signup-grad-year">
                  Graduation Year <span className="auth-required">*</span>
                </label>
                <select
                  id="signup-grad-year"
                  className="auth-input auth-select"
                  value={graduationYear}
                  onChange={e => setGraduationYear(e.target.value)}
                  required
                >
                  {graduationYears.map(y => (
                    <option key={y} value={y}>{y}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="auth-field">
              <label className="auth-label" htmlFor="signup-roll">
                Roll / Enrollment Number <span className="auth-optional">(Optional)</span>
              </label>
              <input
                id="signup-roll"
                type="text"
                className="auth-input"
                placeholder="e.g. 22CS1049"
                value={rollNumber}
                onChange={e => setRollNumber(e.target.value)}
              />
            </div>

            {/* Terms Agreement Checkbox */}
            <div className="auth-terms-group">
              <label className="auth-checkbox-label">
                <input
                  type="checkbox"
                  checked={agreeTerms}
                  onChange={e => setAgreeTerms(e.target.checked)}
                  required
                />
                <span>
                  I agree to the StudySnap AI <span className="auth-terms-link">Terms of Service</span> and <span className="auth-terms-link">Privacy Policy</span>.
                </span>
              </label>
            </div>

            {/* Step 2 Actions */}
            <div className="auth-step-actions">
              <button
                type="button"
                className="auth-back-btn"
                onClick={handlePreviousStep}
                disabled={loading}
              >
                <ArrowLeft size={16} /> Back
              </button>
              <button
                type="submit"
                className="auth-submit-btn auth-submit-final"
                disabled={loading}
              >
                {loading ? (
                  <span className="auth-btn-loader">
                    <Loader2 size={16} className="animate-spin" /> Creating Student Profile...
                  </span>
                ) : (
                  <>
                    <CheckSquare size={16} />
                    Complete Registration
                  </>
                )}
              </button>
            </div>
          </form>
        )}

        {/* Toggle Mode Footer */}
        <div className="auth-toggle-footer">
          {isRegister ? 'Already registered as a student?' : "Don't have a student profile yet?"}{' '}
          <button
            type="button"
            className="auth-toggle-btn"
            onClick={() => {
              setIsRegister(!isRegister);
              setSignupStep(1);
              setErrorMsg(null);
            }}
          >
            {isRegister ? 'Log In here' : 'Sign Up now'}
          </button>
        </div>

        {/* Quick Demo Login Option */}
        <div className="auth-divider">
          <span>OR QUICK EVALUATION</span>
        </div>

        <button
          className="auth-demo-btn"
          onClick={handleQuickDemoLogin}
          disabled={loading}
          type="button"
        >
          <Sparkles size={14} />
          Quick Demo Login
        </button>
      </div>
    </div>
  );
};

export default AuthModal;
