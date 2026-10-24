-- =============================================================================
-- Migration: Add theme_preference to public.profiles and enforce RLS
-- StudySnap AI — VS Code-Inspired Color Theme System
-- =============================================================================

-- 1. Add theme_preference column to public.profiles with default 'dark-modern'
ALTER TABLE IF EXISTS public.profiles 
ADD COLUMN IF NOT EXISTS theme_preference TEXT DEFAULT 'dark-modern';

-- 2. Index for fast theme preference lookups
CREATE INDEX IF NOT EXISTS idx_profiles_theme_preference ON public.profiles(theme_preference);

-- 3. Enforce Row Level Security (RLS) on public.profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policy: Users can only view their own profile
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

-- 5. RLS Policy: Users can only update their own profile and theme_preference
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- 6. RLS Policy: Users can only insert their own profile
DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
CREATE POLICY "Users can insert own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- 7. Update handle_new_user() trigger function to preserve theme_preference on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
DECLARE
    v_full_name TEXT;
    v_college TEXT;
    v_degree TEXT;
    v_custom_degree TEXT;
    v_branch TEXT;
    v_custom_branch TEXT;
    v_academic_year TEXT;
    v_semester TEXT;
    v_graduation_year TEXT;
    v_roll_number TEXT;
    v_avatar_path TEXT;
    v_theme_preference TEXT;
BEGIN
    v_full_name := COALESCE(
        NEW.raw_user_meta_data->>'full_name',
        NEW.raw_user_meta_data->>'name',
        split_part(NEW.email, '@', 1)
    );
    v_college := COALESCE(
        NEW.raw_user_meta_data->>'college_name',
        NEW.raw_user_meta_data->>'college',
        'DKTE'
    );
    v_degree := NEW.raw_user_meta_data->>'degree_program';
    v_custom_degree := NEW.raw_user_meta_data->>'custom_degree';
    v_branch := NEW.raw_user_meta_data->>'branch';
    v_custom_branch := NEW.raw_user_meta_data->>'custom_branch';
    v_academic_year := NEW.raw_user_meta_data->>'academic_year';
    v_semester := NEW.raw_user_meta_data->>'semester';
    v_graduation_year := NEW.raw_user_meta_data->>'graduation_year';
    v_roll_number := NEW.raw_user_meta_data->>'roll_number';
    v_avatar_path := NEW.raw_user_meta_data->>'avatar_path';
    v_theme_preference := COALESCE(NEW.raw_user_meta_data->>'theme_preference', 'dark-modern');

    INSERT INTO public.profiles (
        id,
        full_name,
        display_name,
        college_name,
        college,
        degree_program,
        custom_degree,
        branch,
        custom_branch,
        academic_year,
        semester,
        graduation_year,
        roll_number,
        avatar_path,
        theme_preference,
        updated_at
    )
    VALUES (
        NEW.id,
        v_full_name,
        v_full_name,
        v_college,
        v_college,
        v_degree,
        v_custom_degree,
        v_branch,
        v_custom_branch,
        v_academic_year,
        v_semester,
        v_graduation_year,
        v_roll_number,
        v_avatar_path,
        v_theme_preference,
        timezone('utc'::text, now())
    )
    ON CONFLICT (id) DO UPDATE
    SET full_name = COALESCE(EXCLUDED.full_name, public.profiles.full_name),
        display_name = COALESCE(EXCLUDED.display_name, public.profiles.display_name),
        college_name = COALESCE(EXCLUDED.college_name, public.profiles.college_name),
        college = COALESCE(EXCLUDED.college, public.profiles.college),
        degree_program = COALESCE(EXCLUDED.degree_program, public.profiles.degree_program),
        custom_degree = COALESCE(EXCLUDED.custom_degree, public.profiles.custom_degree),
        branch = COALESCE(EXCLUDED.branch, public.profiles.branch),
        custom_branch = COALESCE(EXCLUDED.custom_branch, public.profiles.custom_branch),
        academic_year = COALESCE(EXCLUDED.academic_year, public.profiles.academic_year),
        semester = COALESCE(EXCLUDED.semester, public.profiles.semester),
        graduation_year = COALESCE(EXCLUDED.graduation_year, public.profiles.graduation_year),
        roll_number = COALESCE(EXCLUDED.roll_number, public.profiles.roll_number),
        avatar_path = COALESCE(EXCLUDED.avatar_path, public.profiles.avatar_path),
        theme_preference = COALESCE(EXCLUDED.theme_preference, public.profiles.theme_preference),
        updated_at = timezone('utc'::text, now());

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
