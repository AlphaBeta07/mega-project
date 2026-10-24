import React, { useState, useEffect, useRef } from 'react';
import {
  X, LogOut, GraduationCap, Mail, ShieldCheck,
  Building2, BookOpen, Calendar, Edit3, Camera,
  CheckCircle2, AlertCircle, Loader2, Save, Undo2,
  KeyRound, Hash, Clock, Trash2, Upload, Image as ImageIcon,
  Palette
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { ThemePickerModal } from './ThemePickerModal';
import {
  updateStudentProfile,
  uploadStudentAvatar,
  removeStudentAvatar,
  validateAvatarFile,
  resetPasswordForEmail,
  fetchStudentProfile
} from '../config/supabase';
import type { StudentProfile } from '../config/supabase';
import {
  DEGREE_OPTIONS,
  BRANCH_OPTIONS,
  ACADEMIC_YEARS,
  getAvailableSemesters,
  getGraduationYears,
  formatStudentProgramBadge
} from '../data/academicData';
import './UserProfileModal.css';

interface UserProfileModalProps {
  isOpen: boolean;
  user: any;
  onClose: () => void;
  onLogout: () => void;
  onProfileUpdated?: (updated: StudentProfile) => void;
}

export const UserProfileModal: React.FC<UserProfileModalProps> = ({
  isOpen,
  user,
  onClose,
  onLogout,
  onProfileUpdated
}) => {
  // Mode: 'view' | 'edit'
  const [isEditing, setIsEditing] = useState(false);
  // Logout confirmation dialog toggle
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  // Remove photo confirmation dialog toggle
  const [showRemovePhotoConfirm, setShowRemovePhotoConfirm] = useState(false);
  // Theme picker dialog toggle
  const [isThemePickerOpen, setIsThemePickerOpen] = useState(false);

  // Theme context
  const { resolvedTheme } = useTheme();

  // Profile data state
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [loadingProfile, setLoadingProfile] = useState(false);

  // Edit form state
  const [editFullName, setEditFullName] = useState('');
  const [editCollege, setEditCollege] = useState('');
  const [editDegree, setEditDegree] = useState('B.Tech / B.E.');
  const [editCustomDegree, setEditCustomDegree] = useState('');
  const [editBranch, setEditBranch] = useState('Computer Science Engineering (CSE)');
  const [editCustomBranch, setEditCustomBranch] = useState('');
  const [editAcademicYear, setEditAcademicYear] = useState('Third Year');
  const [editSemester, setEditSemester] = useState('Semester 5');
  const [editGraduationYear, setEditGraduationYear] = useState('2026');
  const [editRollNumber, setEditRollNumber] = useState('');

  // Status & Actions
  const [saving, setSaving] = useState(false);
  const [uploadingAvatar, setUploadingAvatar] = useState(false);
  const [removingAvatar, setRemovingAvatar] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null);

  // Image Preview State (before saving to Supabase Storage)
  const [pendingAvatarFile, setPendingAvatarFile] = useState<File | null>(null);
  const [pendingAvatarPreview, setPendingAvatarPreview] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Prevent background page scrolling when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Sync profile when opened or user prop changes
  useEffect(() => {
    if (!isOpen) {
      setIsEditing(false);
      setShowLogoutConfirm(false);
      setShowRemovePhotoConfirm(false);
      setStatusMessage(null);
      clearPendingPreview();
      return;
    }

    const loadProfile = async () => {
      setLoadingProfile(true);
      if (user?.id) {
        try {
          const dbProfile = await fetchStudentProfile(user.id);
          if (dbProfile) {
            setProfile(dbProfile);
            syncEditFields(dbProfile);
            setLoadingProfile(false);
            return;
          }
        } catch (err) {
          console.warn('[UserProfileModal] Load profile warning:', err);
        }
      }

      // Fallback to user prop
      const fallback: StudentProfile = {
        id: user?.id || 'current-user',
        full_name: user?.full_name || user?.name || 'Student',
        email: user?.email || '',
        college_name: user?.college_name || user?.college || 'DKTE',
        degree_program: user?.degree_program || 'B.Tech / B.E.',
        custom_degree: user?.custom_degree || null,
        branch: user?.branch || 'Computer Science Engineering (CSE)',
        custom_branch: user?.custom_branch || null,
        academic_year: user?.academic_year || 'Third Year',
        semester: user?.semester || 'Semester 5',
        graduation_year: user?.graduation_year || '2026',
        roll_number: user?.roll_number || null,
        avatar_url: user?.avatar_url || null,
        avatar_path: user?.avatar_path || null,
        email_verified: Boolean(user?.email_verified),
      };
      setProfile(fallback);
      syncEditFields(fallback);
      setLoadingProfile(false);
    };

    loadProfile();
  }, [isOpen, user]);

  const syncEditFields = (p: StudentProfile) => {
    setEditFullName(p.full_name || '');
    setEditCollege(p.college_name || '');
    setEditDegree(p.degree_program || 'B.Tech / B.E.');
    setEditCustomDegree(p.custom_degree || '');
    setEditBranch(p.branch || 'Computer Science Engineering (CSE)');
    setEditCustomBranch(p.custom_branch || '');
    setEditAcademicYear(p.academic_year || 'Third Year');
    setEditSemester(p.semester || 'Semester 5');
    setEditGraduationYear(p.graduation_year || '2026');
    setEditRollNumber(p.roll_number || '');
  };

  const clearPendingPreview = () => {
    if (pendingAvatarPreview && pendingAvatarPreview.startsWith('blob:')) {
      URL.revokeObjectURL(pendingAvatarPreview);
    }
    setPendingAvatarFile(null);
    setPendingAvatarPreview(null);
  };

  // Keyboard accessibility: Escape to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      if (e.key === 'Escape') {
        if (pendingAvatarPreview) {
          clearPendingPreview();
        } else if (showRemovePhotoConfirm) {
          setShowRemovePhotoConfirm(false);
        } else if (showLogoutConfirm) {
          setShowLogoutConfirm(false);
        } else if (isEditing) {
          if (window.confirm('Discard unsaved profile modifications?')) {
            if (profile) syncEditFields(profile);
            setIsEditing(false);
          }
        } else {
          onClose();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isEditing, profile, onClose, pendingAvatarPreview, showRemovePhotoConfirm, showLogoutConfirm]);

  // Dynamic available semesters for edit form
  const availableSemesters = getAvailableSemesters(editAcademicYear);
  const graduationYears = getGraduationYears();

  useEffect(() => {
    if (isEditing && !availableSemesters.includes(editSemester)) {
      setEditSemester(availableSemesters[0] || 'Semester 1');
    }
  }, [editAcademicYear, availableSemesters, editSemester, isEditing]);

  if (!isOpen) return null;

  const currentFullName = profile?.full_name || user?.full_name || user?.name || 'Student';
  const currentEmail = profile?.email || user?.email || 'student@college.edu.in';
  const currentCollege = profile?.college_name || profile?.college || user?.college_name || user?.college || 'DKTE Society\'s Textile and Engineering Institute';
  const currentDegree = profile?.degree_program || user?.degree_program || 'B.Tech / B.E.';
  const currentCustomDegree = profile?.custom_degree || user?.custom_degree;
  const currentBranch = profile?.branch || user?.branch || 'Computer Science Engineering (CSE)';
  const currentCustomBranch = profile?.custom_branch || user?.custom_branch;
  const currentAcademicYear = profile?.academic_year || user?.academic_year || 'Third Year';
  const currentSemester = profile?.semester || user?.semester || 'Semester 5';
  const currentGradYear = profile?.graduation_year || user?.graduation_year || '2026';
  const currentRoll = profile?.roll_number || user?.roll_number;
  const currentAvatarUrl = profile?.avatar_url || user?.avatar_url;
  const isEmailVerified = profile?.email_verified ?? Boolean(user?.email_verified);

  const programBadge = formatStudentProgramBadge(
    currentDegree,
    currentCustomDegree,
    currentBranch,
    currentCustomBranch
  );

  // Avatar initial letter
  const avatarInitial = (currentFullName.trim() ? currentFullName.trim().charAt(0) : 'S').toUpperCase();

  // Handle Avatar Selection from Native File Picker
  const handleAvatarFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate on client layer
    const validation = validateAvatarFile(file);
    if (!validation.valid) {
      setStatusMessage({ type: 'error', text: validation.error || 'Invalid photo file.' });
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    // Set preview mode so user can review and approve before saving
    clearPendingPreview();
    const previewUrl = URL.createObjectURL(file);
    setPendingAvatarFile(file);
    setPendingAvatarPreview(previewUrl);
    setStatusMessage(null);

    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Confirm and Save Selected Avatar to Supabase Storage
  const handleConfirmAvatarSave = async () => {
    if (!pendingAvatarFile || !profile?.id) return;

    setUploadingAvatar(true);
    setStatusMessage(null);

    try {
      const res = await uploadStudentAvatar(profile.id, pendingAvatarFile, profile.avatar_path);
      const updatedProfile: StudentProfile = {
        ...profile,
        avatar_path: res.avatarPath,
        avatar_url: res.avatarUrl,
      };
      setProfile(updatedProfile);
      if (onProfileUpdated) onProfileUpdated(updatedProfile);

      clearPendingPreview();
      setStatusMessage({ type: 'success', text: 'Profile photo updated successfully!' });
    } catch (err: any) {
      console.error('[StudySnap Avatar Upload] Error:', err);
      setStatusMessage({ type: 'error', text: err.message || 'Failed to upload profile photo. Please try again.' });
    } finally {
      setUploadingAvatar(false);
    }
  };

  // Handle Photo Removal
  const handleConfirmRemovePhoto = async () => {
    if (!profile?.id) return;

    setRemovingAvatar(true);
    setStatusMessage(null);

    try {
      await removeStudentAvatar(profile.id, profile.avatar_path);
      const updatedProfile: StudentProfile = {
        ...profile,
        avatar_path: null,
        avatar_url: null,
      };
      setProfile(updatedProfile);
      if (onProfileUpdated) onProfileUpdated(updatedProfile);

      setShowRemovePhotoConfirm(false);
      setStatusMessage({ type: 'success', text: 'Profile photo removed. Avatar reverted to name initials.' });
    } catch (err: any) {
      console.error('[StudySnap Avatar Remove] Error:', err);
      setStatusMessage({ type: 'error', text: err.message || 'Failed to remove profile photo.' });
    } finally {
      setRemovingAvatar(false);
    }
  };

  // Handle Save Profile Details
  const handleSaveProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editFullName.trim()) {
      setStatusMessage({ type: 'error', text: 'Full name cannot be empty.' });
      return;
    }
    if (!editCollege.trim()) {
      setStatusMessage({ type: 'error', text: 'College / university name cannot be empty.' });
      return;
    }

    setSaving(true);
    setStatusMessage(null);

    const updates: Partial<StudentProfile> = {
      full_name: editFullName.trim(),
      display_name: editFullName.trim(),
      college_name: editCollege.trim(),
      college: editCollege.trim(),
      degree_program: editDegree,
      custom_degree: editDegree === 'Other' ? editCustomDegree.trim() : null,
      branch: editBranch,
      custom_branch: (editBranch === 'Other Branch / Department' || editBranch === 'Other') ? editCustomBranch.trim() : null,
      academic_year: editAcademicYear,
      semester: editSemester,
      graduation_year: editGraduationYear,
      roll_number: editRollNumber.trim() || null,
    };

    try {
      if (profile?.id) {
        const saved = await updateStudentProfile(profile.id, updates);
        setProfile(saved);
        if (onProfileUpdated) onProfileUpdated(saved);
      } else {
        const merged = { ...profile, ...updates } as StudentProfile;
        setProfile(merged);
        if (onProfileUpdated) onProfileUpdated(merged);
      }

      setIsEditing(false);
      setStatusMessage({ type: 'success', text: 'Academic profile updated successfully!' });
    } catch (err: any) {
      console.error('[StudySnap Profile Update] Error:', err);
      setStatusMessage({ type: 'error', text: err.message || 'Failed to update profile.' });
    } finally {
      setSaving(false);
    }
  };

  // Handle Password Reset Request
  const handleSendPasswordReset = async () => {
    if (!currentEmail) return;
    try {
      setSaving(true);
      await resetPasswordForEmail(currentEmail);
      setStatusMessage({
        type: 'info',
        text: `Password reset link sent to ${currentEmail}. Please check your email inbox.`
      });
    } catch (err: any) {
      setStatusMessage({ type: 'error', text: err.message || 'Failed to send password reset email.' });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="up-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="up-modal-title">
      <div className="up-modal" onClick={e => e.stopPropagation()}>
        {/* Close Button */}
        <button
          className="up-close"
          onClick={() => {
            if (pendingAvatarPreview) {
              clearPendingPreview();
            } else if (isEditing) {
              if (window.confirm('Discard unsaved profile changes?')) {
                if (profile) syncEditFields(profile);
                setIsEditing(false);
                onClose();
              }
            } else {
              onClose();
            }
          }}
          aria-label="Close profile modal"
        >
          <X size={17} />
        </button>

        {/* ── FIXED MODAL HEADER ── */}
        <header className="up-header">
          {/* Status Alerts */}
          {statusMessage && (
            <div className={`up-status-alert up-status-${statusMessage.type}`} role="alert">
              {statusMessage.type === 'success' && <CheckCircle2 size={15} />}
              {statusMessage.type === 'error' && <AlertCircle size={15} />}
              {statusMessage.type === 'info' && <ShieldCheck size={15} />}
              <span>{statusMessage.text}</span>
            </div>
          )}

          {/* Identity Section */}
          <div className="up-identity-row">
            {/* Avatar with Camera Overlay */}
            <div className="up-avatar-wrap">
              <div
                className="up-avatar"
                onClick={() => fileInputRef.current?.click()}
                role="button"
                tabIndex={0}
                title="Click to upload or change profile photo"
              >
                {currentAvatarUrl ? (
                  <img src={currentAvatarUrl} alt={currentFullName} className="up-avatar-img" />
                ) : (
                  <span className="up-avatar-initial">{avatarInitial}</span>
                )}
                {uploadingAvatar && (
                  <div className="up-avatar-overlay">
                    <Loader2 size={22} className="animate-spin" />
                  </div>
                )}
              </div>

              <button
                type="button"
                className="up-avatar-camera-btn"
                onClick={() => fileInputRef.current?.click()}
                title={currentAvatarUrl ? 'Change profile photo' : 'Upload profile photo'}
                disabled={uploadingAvatar}
                aria-label="Upload photo"
              >
                <Camera size={13} />
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/jpg"
                style={{ display: 'none' }}
                onChange={handleAvatarFileChange}
                disabled={uploadingAvatar}
              />
            </div>

            {/* User Meta Information */}
            <div className="up-identity-meta">
              <div className="up-name-badge-row">
                <h2 id="up-modal-title" className="up-name" title={currentFullName}>
                  {currentFullName}
                </h2>
                <span className={`up-verify-chip ${isEmailVerified ? 'verified' : 'pending'}`}>
                  <ShieldCheck size={12} />
                  {isEmailVerified ? 'Verified' : 'Pending'}
                </span>
              </div>

              <div className="up-email-line">
                <Mail size={13} className="up-email-icon" />
                <span>{currentEmail}</span>
              </div>

              <div className="up-program-chip" title="Academic Program & Branch">
                <GraduationCap size={13} />
                <span>{programBadge}</span>
              </div>

              {/* Contextual Photo Actions */}
              <div className="up-photo-actions-inline">
                {!currentAvatarUrl ? (
                  <button
                    type="button"
                    className="up-photo-text-btn"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploadingAvatar}
                  >
                    <Upload size={12} /> Upload Photo
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      className="up-photo-text-btn"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={uploadingAvatar}
                    >
                      <Camera size={12} /> Change Photo
                    </button>
                    <span className="up-photo-sep">•</span>
                    <button
                      type="button"
                      className="up-photo-text-btn danger"
                      onClick={() => setShowRemovePhotoConfirm(true)}
                      disabled={uploadingAvatar || removingAvatar}
                    >
                      <Trash2 size={12} /> Remove
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        </header>

        {/* ── SCROLLABLE MODAL CONTENT BODY ── */}
        <div className="up-scrollable-body">
          {!isEditing ? (
            /* VIEW MODE */
            loadingProfile ? (
              <div className="up-loading-state">
                <Loader2 size={24} className="animate-spin text-amber-500" />
                <span>Loading student records...</span>
              </div>
            ) : (
              <div className="up-sections-stack">
                {/* ── Section 1: Academic Profile ── */}
                <section className="up-section" aria-labelledby="academic-profile-heading">
                  <div className="up-section-header">
                    <div className="up-section-title-wrap">
                      <GraduationCap size={15} className="up-section-icon" />
                      <h3 id="academic-profile-heading" className="up-section-title">Academic Profile</h3>
                    </div>
                    <span className="up-section-sub">Curriculum & enrollment details</span>
                  </div>

                  <div className="up-academic-grid">
                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <Building2 size={12} /> College / University
                      </span>
                      <span className="up-tile-value" title={currentCollege}>
                        {currentCollege}
                      </span>
                    </div>

                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <BookOpen size={12} /> Degree / Program
                      </span>
                      <span className="up-tile-value" title={currentDegree}>
                        {currentDegree === 'Other' ? (currentCustomDegree || 'Custom Degree') : currentDegree}
                      </span>
                    </div>

                    <div className="up-data-tile up-data-tile-wide">
                      <span className="up-tile-label">
                        <BookOpen size={12} /> Branch / Department
                      </span>
                      <span className="up-tile-value" title={currentBranch}>
                        {(currentBranch === 'Other Branch / Department' || currentBranch === 'Other')
                          ? (currentCustomBranch || 'Custom Branch')
                          : currentBranch}
                      </span>
                    </div>

                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <Calendar size={12} /> Current Academic Year
                      </span>
                      <span className="up-tile-value">
                        {currentAcademicYear}
                      </span>
                    </div>

                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <Clock size={12} /> Current Semester
                      </span>
                      <span className="up-tile-value">
                        {currentSemester}
                      </span>
                    </div>

                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <Calendar size={12} /> Expected Graduation
                      </span>
                      <span className="up-tile-value">
                        Class of {currentGradYear}
                      </span>
                    </div>

                    <div className="up-data-tile">
                      <span className="up-tile-label">
                        <Hash size={12} /> Roll / Enrollment Number
                      </span>
                      <span className="up-tile-value">
                        {currentRoll ? currentRoll : <span className="up-empty-value">Not provided</span>}
                      </span>
                    </div>
                  </div>
                </section>

                {/* ── Section: Appearance & Themes ── */}
                <section className="up-section" aria-labelledby="appearance-themes-heading">
                  <div className="up-section-header">
                    <div className="up-section-title-wrap">
                      <Palette size={15} className="up-section-icon" />
                      <h3 id="appearance-themes-heading" className="up-section-title">Appearance & Themes</h3>
                    </div>
                    <span className="up-section-sub">Customize the colors of your StudySnap AI workspace</span>
                  </div>

                  <div className="up-theme-card">
                    <div className="up-theme-info">
                      <div className="up-theme-name-row">
                        <span className="up-theme-name">{resolvedTheme.name}</span>
                        <span className={`up-theme-badge up-theme-badge-${resolvedTheme.category}`}>
                          {resolvedTheme.category.toUpperCase()}
                        </span>
                      </div>
                      <p className="up-theme-desc">{resolvedTheme.description}</p>
                      <div className="up-theme-swatches-preview">
                        {resolvedTheme.swatches.map((color, idx) => (
                          <span
                            key={idx}
                            className="up-theme-swatch-circle"
                            style={{ backgroundColor: color }}
                            title={`Theme swatch: ${color}`}
                          />
                        ))}
                      </div>
                    </div>

                    <button
                      type="button"
                      className="up-theme-browse-btn"
                      onClick={() => setIsThemePickerOpen(true)}
                      title="Open VS Code-inspired theme selector"
                    >
                      <Palette size={14} /> Browse Themes
                    </button>
                  </div>
                </section>

                {/* ── Section 2: Account & Security ── */}
                <section className="up-section" aria-labelledby="account-security-heading">
                  <div className="up-section-header">
                    <div className="up-section-title-wrap">
                      <KeyRound size={15} className="up-section-icon" />
                      <h3 id="account-security-heading" className="up-section-title">Account & Security</h3>
                    </div>
                    <span className="up-section-sub">Authentication and credentials</span>
                  </div>

                  <div className="up-security-card">
                    <div className="up-security-info">
                      <div className="up-security-label">Primary Account Email</div>
                      <div className="up-security-val">{currentEmail}</div>
                      <div className="up-security-meta">
                        {isEmailVerified ? (
                          <span className="up-verified-text">
                            {/* <CheckCircle2 size={12} /> Verified via Supabase Auth */}
                          </span>
                        ) : (
                          <span className="up-pending-text">
                            <AlertCircle size={12} /> Email verification email pending
                          </span>
                        )}
                      </div>
                    </div>

                    <button
                      type="button"
                      className="up-password-btn"
                      onClick={handleSendPasswordReset}
                      disabled={saving}
                      title="Send password recovery instructions to email"
                    >
                      <KeyRound size={13} /> Change Password
                    </button>
                  </div>
                </section>
              </div>
            )
          ) : (
            /* EDIT MODE FORM */
            <form id="up-profile-form" className="up-edit-form" onSubmit={handleSaveProfile}>
              <div className="up-edit-notice">
                <span>Updating your academic records synchronizes your active courses and lecture notes with Supabase.</span>
              </div>

              <div className="up-form-grid-2">
                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-name">
                    Full Name <span className="up-req">*</span>
                  </label>
                  <input
                    id="edit-name"
                    type="text"
                    className="up-form-input"
                    value={editFullName}
                    onChange={e => setEditFullName(e.target.value)}
                    placeholder="e.g. Rohan Sharma"
                    required
                  />
                </div>

                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-college">
                    College / University Name <span className="up-req">*</span>
                  </label>
                  <input
                    id="edit-college"
                    type="text"
                    className="up-form-input"
                    value={editCollege}
                    onChange={e => setEditCollege(e.target.value)}
                    placeholder="e.g. DKTE Society's Textile & Engineering Institute"
                    required
                  />
                </div>
              </div>

              <div className="up-form-grid-2">
                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-degree">
                    Degree / Program <span className="up-req">*</span>
                  </label>
                  <select
                    id="edit-degree"
                    className="up-form-input up-select"
                    value={editDegree}
                    onChange={e => setEditDegree(e.target.value)}
                    required
                  >
                    {DEGREE_OPTIONS.map(d => (
                      <option key={d.value} value={d.value}>{d.label}</option>
                    ))}
                  </select>
                </div>

                {editDegree === 'Other' && (
                  <div className="up-form-group">
                    <label className="up-form-label" htmlFor="edit-custom-degree">
                      Custom Degree <span className="up-req">*</span>
                    </label>
                    <input
                      id="edit-custom-degree"
                      type="text"
                      className="up-form-input"
                      value={editCustomDegree}
                      onChange={e => setEditCustomDegree(e.target.value)}
                      placeholder="Enter degree name"
                      required
                    />
                  </div>
                )}

                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-branch">
                    Branch / Department <span className="up-req">*</span>
                  </label>
                  <select
                    id="edit-branch"
                    className="up-form-input up-select"
                    value={editBranch}
                    onChange={e => setEditBranch(e.target.value)}
                    required
                  >
                    {BRANCH_OPTIONS.map(b => (
                      <option key={b} value={b}>{b}</option>
                    ))}
                  </select>
                </div>

                {(editBranch === 'Other Branch / Department' || editBranch === 'Other') && (
                  <div className="up-form-group">
                    <label className="up-form-label" htmlFor="edit-custom-branch">
                      Custom Branch <span className="up-req">*</span>
                    </label>
                    <input
                      id="edit-custom-branch"
                      type="text"
                      className="up-form-input"
                      value={editCustomBranch}
                      onChange={e => setEditCustomBranch(e.target.value)}
                      placeholder="Enter branch / field"
                      required
                    />
                  </div>
                )}
              </div>

              <div className="up-form-grid-3">
                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-year">
                    Academic Year <span className="up-req">*</span>
                  </label>
                  <select
                    id="edit-year"
                    className="up-form-input up-select"
                    value={editAcademicYear}
                    onChange={e => setEditAcademicYear(e.target.value)}
                    required
                  >
                    {ACADEMIC_YEARS.map(y => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>

                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-sem">
                    Semester <span className="up-req">*</span>
                  </label>
                  <select
                    id="edit-sem"
                    className="up-form-input up-select"
                    value={editSemester}
                    onChange={e => setEditSemester(e.target.value)}
                    required
                  >
                    {availableSemesters.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>

                <div className="up-form-group">
                  <label className="up-form-label" htmlFor="edit-grad">
                    Graduation Year <span className="up-req">*</span>
                  </label>
                  <select
                    id="edit-grad"
                    className="up-form-input up-select"
                    value={editGraduationYear}
                    onChange={e => setEditGraduationYear(e.target.value)}
                    required
                  >
                    {graduationYears.map(y => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="up-form-group">
                <label className="up-form-label" htmlFor="edit-roll">
                  Roll / Enrollment Number <span className="up-opt">(Optional)</span>
                </label>
                <input
                  id="edit-roll"
                  type="text"
                  className="up-form-input"
                  placeholder="e.g. 22CS1049"
                  value={editRollNumber}
                  onChange={e => setEditRollNumber(e.target.value)}
                />
              </div>
            </form>
          )}
        </div>

        {/* ── STICKY MODAL FOOTER (Never clipped) ── */}
        <footer className="up-modal-footer">
          {!isEditing ? (
            <>
              <button
                type="button"
                className="up-btn-secondary danger"
                onClick={() => setShowLogoutConfirm(true)}
              >
                <LogOut size={14} /> Log Out
              </button>

              <button
                type="button"
                className="up-btn-primary"
                onClick={() => {
                  if (profile) syncEditFields(profile);
                  setIsEditing(true);
                  setStatusMessage(null);
                }}
              >
                <Edit3 size={14} /> Edit Academic Profile
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="up-btn-secondary"
                onClick={() => {
                  if (profile) syncEditFields(profile);
                  setIsEditing(false);
                  setStatusMessage(null);
                }}
                disabled={saving}
              >
                <Undo2 size={14} /> Cancel
              </button>

              <button
                type="submit"
                form="up-profile-form"
                className="up-btn-primary"
                disabled={saving}
              >
                {saving ? (
                  <>
                    <Loader2 size={14} className="animate-spin" /> Saving Changes...
                  </>
                ) : (
                  <>
                    <Save size={14} /> Save Changes
                  </>
                )}
              </button>
            </>
          )}
        </footer>

        {/* ── PHOTO PREVIEW MODAL ── */}
        {pendingAvatarPreview && pendingAvatarFile && (
          <div className="up-submodal-overlay" onClick={clearPendingPreview}>
            <div className="up-preview-dialog" onClick={e => e.stopPropagation()}>
              <div className="up-preview-header">
                <ImageIcon size={18} className="text-amber-400" />
                <h3 className="up-preview-title">Preview Profile Photo</h3>
              </div>

              <p className="up-preview-desc">
                Review how your photo will appear across your student profile and dashboard header.
              </p>

              <div className="up-preview-circle">
                <img src={pendingAvatarPreview} alt="Avatar preview" className="up-preview-img" />
              </div>

              <div className="up-preview-meta">
                <span className="up-preview-filename">{pendingAvatarFile.name}</span>
                <span className="up-preview-filesize">
                  {(pendingAvatarFile.size / (1024 * 1024)).toFixed(2)} MB • {pendingAvatarFile.type.replace('image/', '').toUpperCase()}
                </span>
              </div>

              <div className="up-preview-actions">
                <button
                  type="button"
                  className="up-sub-cancel-btn"
                  onClick={clearPendingPreview}
                  disabled={uploadingAvatar}
                >
                  Cancel
                </button>

                <button
                  type="button"
                  className="up-sub-change-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploadingAvatar}
                >
                  Choose Different
                </button>

                <button
                  type="button"
                  className="up-sub-save-btn"
                  onClick={handleConfirmAvatarSave}
                  disabled={uploadingAvatar}
                >
                  {uploadingAvatar ? (
                    <>
                      <Loader2 size={14} className="animate-spin" /> Saving...
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={14} /> Save & Apply
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── REMOVE PHOTO CONFIRMATION DIALOG ── */}
        {showRemovePhotoConfirm && (
          <div className="up-submodal-overlay" onClick={() => setShowRemovePhotoConfirm(false)}>
            <div className="up-confirm-dialog" onClick={e => e.stopPropagation()}>
              <div className="up-confirm-icon up-confirm-danger">
                <Trash2 size={20} />
              </div>
              <h3 className="up-confirm-title">Remove Profile Photo?</h3>
              <p className="up-confirm-desc">
                Your uploaded profile picture will be permanently deleted from storage. Your avatar will revert to your name initials (<strong style={{ color: '#FBBF24' }}>{avatarInitial}</strong>).
              </p>
              <div className="up-confirm-buttons">
                <button
                  type="button"
                  className="up-sub-cancel-btn"
                  onClick={() => setShowRemovePhotoConfirm(false)}
                  disabled={removingAvatar}
                >
                  Keep Photo
                </button>
                <button
                  type="button"
                  className="up-sub-danger-btn"
                  onClick={handleConfirmRemovePhoto}
                  disabled={removingAvatar}
                >
                  {removingAvatar ? (
                    <>
                      <Loader2 size={14} className="animate-spin" /> Removing...
                    </>
                  ) : (
                    'Yes, Remove Photo'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── LOGOUT CONFIRMATION MODAL ── */}
        {showLogoutConfirm && (
          <div className="up-submodal-overlay" onClick={() => setShowLogoutConfirm(false)}>
            <div className="up-confirm-dialog" onClick={e => e.stopPropagation()}>
              <div className="up-confirm-icon">
                <LogOut size={20} />
              </div>
              <h3 className="up-confirm-title">Confirm Sign Out</h3>
              <p className="up-confirm-desc">
                Are you sure you want to log out of StudySnap AI? Your active notebooks and saved study sessions will remain secure in Supabase.
              </p>
              <div className="up-confirm-buttons">
                <button
                  type="button"
                  className="up-sub-cancel-btn"
                  onClick={() => setShowLogoutConfirm(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="up-sub-danger-btn"
                  onClick={() => {
                    setShowLogoutConfirm(false);
                    onLogout();
                    onClose();
                  }}
                >
                  Log Out
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Theme Picker Modal ── */}
        <ThemePickerModal
          isOpen={isThemePickerOpen}
          onClose={() => setIsThemePickerOpen(false)}
          onThemeSaved={(savedThemeId) => {
            const currentP = profile || user;
            if (currentP) {
              const updated = { ...currentP, theme_preference: savedThemeId };
              setProfile(updated as StudentProfile);
              onProfileUpdated?.(updated as StudentProfile);
            }
          }}
        />
      </div>
    </div>
  );
};

export default UserProfileModal;
