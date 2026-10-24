import { createClient, SupabaseClient } from '@supabase/supabase-js';
import type { User, Session } from '@supabase/supabase-js';
import { setAuthToken } from './api';

// Cleanly resolve Supabase URL and Anon Key (prioritize standard JWT anon key)
const supabaseUrl = (import.meta.env.VITE_SUPABASE_URL || '').trim();
const supabaseAnonKey = (
  import.meta.env.VITE_SUPABASE_ANON_KEY ||
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY ||
  ''
).trim();

const isValidUrl = (url: string) => {
  try {
    return Boolean(url && new URL(url).protocol.startsWith('http') && !url.includes('your-project-id'));
  } catch {
    return false;
  }
};

export const isSupabaseConfigured = (): boolean => {
  return isValidUrl(supabaseUrl) && Boolean(
    supabaseAnonKey &&
    !supabaseAnonKey.includes('your-anon-key') &&
    supabaseAnonKey.length > 20
  );
};

let clientInstance: SupabaseClient | null = null;

if (isSupabaseConfigured()) {
  try {
    clientInstance = createClient(supabaseUrl, supabaseAnonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true,
        storage: window.localStorage,
        storageKey: 'studysnap_supabase_auth',
      },
    });

    // Automatically sync Supabase session token to API client
    clientInstance.auth.getSession().then(({ data }) => {
      if (data?.session?.access_token) {
        setAuthToken(data.session.access_token);
      }
    });

    // Listen to Supabase auth state transitions
    clientInstance.auth.onAuthStateChange(async (_event, session) => {
      if (session?.access_token) {
        setAuthToken(session.access_token);
        if (session.user) {
          const profile = await fetchStudentProfile(session.user.id).catch(() => null);
          const userMeta = {
            id: session.user.id,
            email: session.user.email || '',
            name: profile?.full_name || session.user.user_metadata?.full_name || session.user.user_metadata?.name || session.user.email?.split('@')[0] || 'Student',
            full_name: profile?.full_name || session.user.user_metadata?.full_name || session.user.user_metadata?.name || 'Student',
            college: profile?.college_name || session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
            college_name: profile?.college_name || session.user.user_metadata?.college_name || session.user.user_metadata?.college || 'DKTE',
            degree_program: profile?.degree_program || session.user.user_metadata?.degree_program || 'B.Tech / B.E.',
            custom_degree: profile?.custom_degree || session.user.user_metadata?.custom_degree || '',
            branch: profile?.branch || session.user.user_metadata?.branch || 'Computer Science Engineering (CSE)',
            custom_branch: profile?.custom_branch || session.user.user_metadata?.custom_branch || '',
            academic_year: profile?.academic_year || session.user.user_metadata?.academic_year || 'Third Year',
            semester: profile?.semester || session.user.user_metadata?.semester || 'Semester 5',
            graduation_year: profile?.graduation_year || session.user.user_metadata?.graduation_year || '2026',
            roll_number: profile?.roll_number || session.user.user_metadata?.roll_number || '',
            avatar_url: profile?.avatar_url || session.user.user_metadata?.avatar_url || null,
            avatar_path: profile?.avatar_path || session.user.user_metadata?.avatar_path || null,
            email_verified: Boolean(session.user.email_confirmed_at),
          };
          localStorage.setItem('studysnap_user', JSON.stringify(userMeta));
        }
      } else {
        setAuthToken(null);
        localStorage.removeItem('studysnap_user');
      }
    });
  } catch (err) {
    console.warn('[StudySnap Supabase] Initialization warning:', err);
  }
} else {
  console.info('[StudySnap Supabase] Supabase is not configured yet. Configure VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env');
}

export const supabase: SupabaseClient | null = clientInstance;

// =============================================================================
// Student Profile & Signup Interfaces
// =============================================================================
export interface StudentProfile {
  id: string;
  full_name: string;
  display_name?: string;
  email?: string;
  college_name: string;
  college?: string;
  degree_program: string;
  custom_degree?: string | null;
  branch: string;
  custom_branch?: string | null;
  academic_year: string;
  semester: string;
  graduation_year: string;
  roll_number?: string | null;
  avatar_path?: string | null;
  avatar_url?: string | null;
  theme_preference?: string | null;
  created_at?: string;
  updated_at?: string;
  email_verified?: boolean;
}

export interface StudentSignupData {
  fullName: string;
  email: string;
  password: string;
  collegeName: string;
  degreeProgram: string;
  customDegree?: string;
  branch: string;
  customBranch?: string;
  academicYear: string;
  semester: string;
  graduationYear: string;
  rollNumber?: string;
}

/**
 * Register a student account with Supabase Auth and save their academic profile
 */
export const signUpStudent = async (data: StudentSignupData) => {
  if (!supabase) {
    throw new Error('Supabase is not configured. Please add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env');
  }

  const { data: authData, error: authError } = await supabase.auth.signUp({
    email: data.email.trim(),
    password: data.password,
    options: {
      data: {
        name: data.fullName.trim(),
        full_name: data.fullName.trim(),
        college: data.collegeName.trim(),
        college_name: data.collegeName.trim(),
        degree_program: data.degreeProgram,
        custom_degree: data.customDegree?.trim() || null,
        branch: data.branch,
        custom_branch: data.customBranch?.trim() || null,
        academic_year: data.academicYear,
        semester: data.semester,
        graduation_year: data.graduationYear,
        roll_number: data.rollNumber?.trim() || null,
      },
    },
  });

  if (authError) throw authError;

  // If user got an immediate session (e.g. email confirmation disabled), update profile table directly
  if (authData.user && authData.session) {
    try {
      const now = new Date().toISOString();
      await supabase.from('profiles').upsert({
        id: authData.user.id,
        full_name: data.fullName.trim(),
        display_name: data.fullName.trim(),
        college_name: data.collegeName.trim(),
        college: data.collegeName.trim(),
        degree_program: data.degreeProgram,
        custom_degree: data.customDegree?.trim() || null,
        branch: data.branch,
        custom_branch: data.customBranch?.trim() || null,
        academic_year: data.academicYear,
        semester: data.semester,
        graduation_year: data.graduationYear,
        roll_number: data.rollNumber?.trim() || null,
        updated_at: now,
      });
    } catch (upsertErr) {
      console.warn('[StudySnap Supabase] Profile direct upsert notice:', upsertErr);
    }
  }

  return authData;
};

/**
 * Legacy signup method for backwards compatibility
 */
export const signUpWithEmail = async (email: string, password: string, name: string, college: string) => {
  return signUpStudent({
    fullName: name,
    email,
    password,
    collegeName: college,
    degreeProgram: 'B.Tech / B.E.',
    branch: 'Computer Science Engineering (CSE)',
    academicYear: 'Third Year',
    semester: 'Semester 5',
    graduationYear: (new Date().getFullYear() + 1).toString(),
  });
};

export const signInWithEmail = async (email: string, password: string) => {
  if (!supabase) {
    throw new Error('Supabase is not configured. Please add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env');
  }
  return await supabase.auth.signInWithPassword({
    email: email.trim(),
    password,
  });
};

export const signOutUser = async () => {
  if (supabase) {
    await supabase.auth.signOut();
  }
  setAuthToken(null);
  localStorage.removeItem('studysnap_user');
};

export const getCurrentSession = async (): Promise<Session | null> => {
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session;
};

export const getCurrentUser = async (): Promise<User | null> => {
  if (!supabase) return null;
  const { data } = await supabase.auth.getUser();
  return data.user;
};

export const resetPasswordForEmail = async (email: string) => {
  if (!supabase) {
    throw new Error('Supabase is not configured. Please add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to frontend/.env');
  }
  return await supabase.auth.resetPasswordForEmail(email.trim(), {
    redirectTo: `${window.location.origin}?reset=true`,
  });
};

// =============================================================================
// Student Profile Management & Avatar Storage
// =============================================================================

/**
 * Fetch student profile from Supabase with fallback to auth metadata
 */
export const fetchStudentProfile = async (userId: string): Promise<StudentProfile | null> => {
  if (!supabase) return null;

  try {
    const { data, error } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .maybeSingle();

    if (error) {
      console.warn('[StudySnap Supabase] Fetch profile notice:', error.message);
    }

    const authUser = await getCurrentUser();
    const meta = authUser?.user_metadata || {};

    let avatarUrl = data?.avatar_url || null;
    if (!avatarUrl && data?.avatar_path) {
      avatarUrl = getStudentAvatarUrl(data.avatar_path);
    }

    return {
      id: userId,
      full_name: data?.full_name || data?.display_name || meta.full_name || meta.name || authUser?.email?.split('@')[0] || 'Student',
      display_name: data?.display_name || data?.full_name || meta.name || 'Student',
      email: authUser?.email || '',
      college_name: data?.college_name || data?.college || meta.college_name || meta.college || 'DKTE',
      college: data?.college || data?.college_name || meta.college || 'DKTE',
      degree_program: data?.degree_program || meta.degree_program || 'B.Tech / B.E.',
      custom_degree: data?.custom_degree || meta.custom_degree || null,
      branch: data?.branch || meta.branch || 'Computer Science Engineering (CSE)',
      custom_branch: data?.custom_branch || meta.custom_branch || null,
      academic_year: data?.academic_year || meta.academic_year || 'Third Year',
      semester: data?.semester || meta.semester || 'Semester 5',
      graduation_year: data?.graduation_year || meta.graduation_year || (new Date().getFullYear() + 1).toString(),
      roll_number: data?.roll_number || meta.roll_number || null,
      avatar_path: data?.avatar_path || null,
      avatar_url: avatarUrl,
      theme_preference: data?.theme_preference || meta.theme_preference || 'dark-modern',
      created_at: data?.created_at,
      updated_at: data?.updated_at,
      email_verified: Boolean(authUser?.email_confirmed_at),
    };
  } catch (err) {
    console.warn('[StudySnap Supabase] Profile retrieval fallback:', err);
    return null;
  }
};

/**
 * Update student profile in Supabase PostgreSQL
 */
export const updateStudentProfile = async (
  userId: string,
  updates: Partial<StudentProfile>
): Promise<StudentProfile> => {
  if (!supabase) throw new Error('Supabase is not configured');

  const now = new Date().toISOString();
  const dbPayload: any = {
    updated_at: now,
  };

  if (updates.full_name !== undefined) {
    dbPayload.full_name = updates.full_name;
    dbPayload.display_name = updates.full_name;
  }
  if (updates.college_name !== undefined) {
    dbPayload.college_name = updates.college_name;
    dbPayload.college = updates.college_name;
  }
  if (updates.degree_program !== undefined) dbPayload.degree_program = updates.degree_program;
  if (updates.custom_degree !== undefined) dbPayload.custom_degree = updates.custom_degree;
  if (updates.branch !== undefined) dbPayload.branch = updates.branch;
  if (updates.custom_branch !== undefined) dbPayload.custom_branch = updates.custom_branch;
  if (updates.academic_year !== undefined) dbPayload.academic_year = updates.academic_year;
  if (updates.semester !== undefined) dbPayload.semester = updates.semester;
  if (updates.graduation_year !== undefined) dbPayload.graduation_year = updates.graduation_year;
  if (updates.roll_number !== undefined) dbPayload.roll_number = updates.roll_number;
  if (updates.avatar_path !== undefined) dbPayload.avatar_path = updates.avatar_path;
  if (updates.avatar_url !== undefined) dbPayload.avatar_url = updates.avatar_url;
  if (updates.theme_preference !== undefined) dbPayload.theme_preference = updates.theme_preference;

  const { error } = await supabase
    .from('profiles')
    .upsert({
      id: userId,
      ...dbPayload,
    })
    .select()
    .single();

  if (error) {
    console.error('[StudySnap Supabase] Update profile error:', error.message);
    throw error;
  }

  // Update Auth user metadata in parallel
  try {
    await supabase.auth.updateUser({
      data: {
        name: dbPayload.full_name,
        full_name: dbPayload.full_name,
        college: dbPayload.college_name,
        college_name: dbPayload.college_name,
        degree_program: dbPayload.degree_program,
        custom_degree: dbPayload.custom_degree,
        branch: dbPayload.branch,
        custom_branch: dbPayload.custom_branch,
        academic_year: dbPayload.academic_year,
        semester: dbPayload.semester,
        graduation_year: dbPayload.graduation_year,
        roll_number: dbPayload.roll_number,
        avatar_path: dbPayload.avatar_path,
        avatar_url: dbPayload.avatar_url,
        theme_preference: dbPayload.theme_preference,
      },
    });
  } catch (metaErr) {
    console.warn('[StudySnap Supabase] Auth metadata sync warning:', metaErr);
  }

  const refreshed = await fetchStudentProfile(userId);
  return refreshed || { ...updates, id: userId } as StudentProfile;
};

/**
 * Validate avatar file format and size (max 5MB, JPEG/PNG/WebP)
 */
export const validateAvatarFile = (file: File): { valid: boolean; error?: string } => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
  if (!allowedTypes.includes(file.type.toLowerCase())) {
    return {
      valid: false,
      error: `Unsupported file format (${file.type || 'unknown'}). Please upload a JPEG, PNG, or WebP image.`
    };
  }

  const maxSizeBytes = 5 * 1024 * 1024; // 5 MB
  if (file.size > maxSizeBytes) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return {
      valid: false,
      error: `File size (${sizeMb} MB) exceeds the maximum limit of 5 MB. Please choose a smaller photo.`
    };
  }

  return { valid: true };
};

/**
 * Upload student avatar image to Supabase Storage with ownership isolation & safe replacement
 */
export const uploadStudentAvatar = async (
  userId: string,
  file: File,
  previousAvatarPath?: string | null
): Promise<{ avatarPath: string; avatarUrl: string }> => {
  // Validate on client layer first
  const validation = validateAvatarFile(file);
  if (!validation.valid) {
    throw new Error(validation.error || 'Invalid photo file');
  }

  if (!supabase) {
    // Offline / fallback demo mode: generate a local object URL
    const localUrl = URL.createObjectURL(file);
    const mockPath = `avatars:${userId}/avatar_${Date.now()}`;
    await updateStudentProfile(userId, {
      avatar_path: mockPath,
      avatar_url: localUrl,
    });
    return {
      avatarPath: mockPath,
      avatarUrl: localUrl,
    };
  }

  // Derive clean extension
  let ext = 'webp';
  if (file.type === 'image/png') ext = 'png';
  else if (file.type === 'image/jpeg' || file.type === 'image/jpg') ext = 'jpg';

  const cleanFileName = `avatar_${Date.now()}.${ext}`;
  const path = `${userId}/${cleanFileName}`;

  // Try 'avatars' bucket first, fallback to 'studysnap-assets'
  let targetBucket = 'avatars';
  let uploadRes = await supabase.storage.from(targetBucket).upload(path, file, {
    contentType: file.type,
    cacheControl: '3600',
    upsert: true,
  });

  if (uploadRes.error) {
    console.warn('[StudySnap Supabase] Avatars bucket upload notice, trying studysnap-assets:', uploadRes.error.message);
    targetBucket = 'studysnap-assets';
    const fallbackPath = `${userId}/avatars/${cleanFileName}`;
    uploadRes = await supabase.storage.from(targetBucket).upload(fallbackPath, file, {
      contentType: file.type,
      cacheControl: '3600',
      upsert: true,
    });
    if (uploadRes.error) {
      throw uploadRes.error;
    }
  }

  const storagePath = uploadRes.data.path;
  const { data: publicUrlData } = supabase.storage.from(targetBucket).getPublicUrl(storagePath);
  let avatarUrl = publicUrlData.publicUrl;

  // Add cache buster timestamp to ensure immediate browser refresh
  if (avatarUrl && !avatarUrl.includes('?')) {
    avatarUrl = `${avatarUrl}?t=${Date.now()}`;
  }

  const fullAvatarPath = `${targetBucket}:${storagePath}`;

  // Persist storage path in profiles table
  await updateStudentProfile(userId, {
    avatar_path: fullAvatarPath,
    avatar_url: avatarUrl,
  });

  // Clean up previous avatar file to prevent orphaned storage objects
  if (previousAvatarPath && previousAvatarPath !== fullAvatarPath) {
    try {
      let prevBucket = 'avatars';
      let prevPath = previousAvatarPath;
      if (previousAvatarPath.includes(':')) {
        const parts = previousAvatarPath.split(':');
        prevBucket = parts[0];
        prevPath = parts.slice(1).join(':');
      }
      await supabase.storage.from(prevBucket).remove([prevPath]);
    } catch (cleanupErr) {
      console.warn('[StudySnap Supabase] Previous avatar cleanup notice:', cleanupErr);
    }
  }

  return {
    avatarPath: fullAvatarPath,
    avatarUrl,
  };
};

/**
 * Remove student profile photo from Supabase Storage and reset profile
 */
export const removeStudentAvatar = async (
  userId: string,
  currentAvatarPath?: string | null
): Promise<{ success: boolean }> => {
  if (supabase && currentAvatarPath) {
    try {
      let bucket = 'avatars';
      let path = currentAvatarPath;
      if (currentAvatarPath.includes(':')) {
        const parts = currentAvatarPath.split(':');
        bucket = parts[0];
        path = parts.slice(1).join(':');
      }
      await supabase.storage.from(bucket).remove([path]);
    } catch (storageErr) {
      console.warn('[StudySnap Supabase] Avatar file removal notice:', storageErr);
    }
  }

  // Clear avatar fields in profiles database table
  await updateStudentProfile(userId, {
    avatar_path: null,
    avatar_url: null,
  });

  return { success: true };
};

/**
 * Resolve avatar path to browser displayable URL
 */
export const getStudentAvatarUrl = (avatarPath: string): string | null => {
  if (!avatarPath || !supabase) return null;
  if (avatarPath.startsWith('http://') || avatarPath.startsWith('https://') || avatarPath.startsWith('data:') || avatarPath.startsWith('blob:')) {
    return avatarPath;
  }

  try {
    let bucket = 'avatars';
    let path = avatarPath;
    if (avatarPath.includes(':')) {
      const parts = avatarPath.split(':');
      bucket = parts[0];
      path = parts.slice(1).join(':');
    }
    const { data } = supabase.storage.from(bucket).getPublicUrl(path);
    return data.publicUrl;
  } catch {
    return null;
  }
};

// =============================================================================
// Notebook Records CRUD
// =============================================================================
export interface SupabaseNotebookRecord {
  id: string;
  owner_id: string;
  user_id?: string;
  title: string;
  description?: string;
  code?: string;
  content?: string;
  sources_count?: number;
  notes_count?: number;
  status?: string;
  created_at: string;
  updated_at: string;
}

export const fetchNotebooksFromSupabase = async (ownerId: string): Promise<SupabaseNotebookRecord[]> => {
  if (!supabase) return [];
  const { data, error } = await supabase
    .from('notebooks')
    .select('*')
    .eq('owner_id', ownerId)
    .order('updated_at', { ascending: false });

  if (error) {
    console.warn('[StudySnap Supabase] Fetch notebooks error:', error.message);
    throw error;
  }
  return (data || []).map((nb: any) => ({
    ...nb,
    user_id: nb.owner_id,
  }));
};

export const createNotebookInSupabase = async (
  ownerId: string,
  title: string = 'Untitled Notebook',
  code: string = 'CS101'
): Promise<SupabaseNotebookRecord> => {
  if (!supabase) throw new Error('Supabase is not configured');
  const id = crypto.randomUUID();
  const now = new Date().toISOString();
  const { data, error } = await supabase
    .from('notebooks')
    .insert([
      {
        id,
        owner_id: ownerId,
        title,
        code,
        content: '',
        sources_count: 0,
        notes_count: 0,
        status: 'ready',
        created_at: now,
        updated_at: now,
      },
    ])
    .select()
    .single();

  if (error) {
    console.warn('[StudySnap Supabase] Create notebook error:', error.message);
    throw error;
  }
  return { ...data, user_id: data.owner_id };
};

export const updateNotebookInSupabase = async (
  id: string,
  ownerId: string,
  updates: Partial<SupabaseNotebookRecord>
): Promise<SupabaseNotebookRecord> => {
  if (!supabase) throw new Error('Supabase is not configured');
  const now = new Date().toISOString();
  const cleanUpdates: any = { ...updates, updated_at: now };
  delete cleanUpdates.id;
  delete cleanUpdates.user_id;
  delete cleanUpdates.owner_id;

  const { data, error } = await supabase
    .from('notebooks')
    .update(cleanUpdates)
    .eq('id', id)
    .eq('owner_id', ownerId)
    .select()
    .single();

  if (error) {
    console.warn('[StudySnap Supabase] Update notebook error:', error.message);
    throw error;
  }
  return { ...data, user_id: data.owner_id };
};

export const deleteNotebookFromSupabase = async (id: string, ownerId: string): Promise<boolean> => {
  if (!supabase) return false;
  const { error } = await supabase
    .from('notebooks')
    .delete()
    .eq('id', id)
    .eq('owner_id', ownerId);

  if (error) {
    console.warn('[StudySnap Supabase] Delete notebook error:', error.message);
    throw error;
  }
  return true;
};
