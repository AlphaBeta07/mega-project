-- =============================================================================
-- StudySnap AI — Student Profile Schema Upgrade Migration
-- Migration: 20261024000001_student_profile_upgrade.sql
-- 100% Idempotent and backwards-compatible with existing profiles
-- =============================================================================

-- 1. Ensure extensions exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Add full student academic columns to public.profiles table
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS full_name TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS college_name TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS degree_program TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS custom_degree TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS branch TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS custom_branch TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS academic_year TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS semester TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS graduation_year TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS roll_number TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS avatar_path TEXT;
ALTER TABLE IF EXISTS public.profiles ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- 3. Backfill existing profile rows for backwards compatibility
UPDATE public.profiles
SET full_name = display_name
WHERE full_name IS NULL AND display_name IS NOT NULL;

UPDATE public.profiles
SET display_name = full_name
WHERE display_name IS NULL AND full_name IS NOT NULL;

UPDATE public.profiles
SET college_name = college
WHERE college_name IS NULL AND college IS NOT NULL;

UPDATE public.profiles
SET college = college_name
WHERE college IS NULL AND college_name IS NOT NULL;

-- 4. Bidirectional synchronization trigger for legacy and modern columns
CREATE OR REPLACE FUNCTION public.sync_profile_legacy_columns()
RETURNS TRIGGER AS $$
BEGIN
    -- Sync full_name <-> display_name
    IF NEW.full_name IS NOT NULL AND NEW.display_name IS NULL THEN
        NEW.display_name = NEW.full_name;
    ELSIF NEW.display_name IS NOT NULL AND NEW.full_name IS NULL THEN
        NEW.full_name = NEW.display_name;
    END IF;

    -- Sync college_name <-> college
    IF NEW.college_name IS NOT NULL AND NEW.college IS NULL THEN
        NEW.college = NEW.college_name;
    ELSIF NEW.college IS NOT NULL AND NEW.college_name IS NULL THEN
        NEW.college_name = NEW.college;
    END IF;

    -- Always refresh updated_at
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_sync_profile_columns ON public.profiles;
CREATE TRIGGER trigger_sync_profile_columns
    BEFORE INSERT OR UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION public.sync_profile_legacy_columns();

-- 5. Enhanced handle_new_user() trigger for auth.users
-- Automatically extracts student academic details from raw_user_meta_data
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
        updated_at = timezone('utc'::text, now());

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Re-attach trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 6. Indexes for academic query performance
CREATE INDEX IF NOT EXISTS idx_profiles_college ON public.profiles(college_name);
CREATE INDEX IF NOT EXISTS idx_profiles_degree ON public.profiles(degree_program);
CREATE INDEX IF NOT EXISTS idx_profiles_branch ON public.profiles(branch);
CREATE INDEX IF NOT EXISTS idx_profiles_grad_year ON public.profiles(graduation_year);

-- 7. Ensure RLS is enabled and policies are watertight
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
CREATE POLICY "Users can insert own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- 8. Storage bucket for user avatars
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'avatars',
    'avatars',
    true,
    5242880, -- 5MB limit
    ARRAY['image/png', 'image/jpeg', 'image/webp', 'image/gif']
)
ON CONFLICT (id) DO UPDATE
SET public = true,
    file_size_limit = 5242880;

-- Storage policies for avatars (user-scoped folder: avatars/<user_id>/<filename>)
DROP POLICY IF EXISTS "Public can view avatars" ON storage.objects;
CREATE POLICY "Public can view avatars"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'avatars');

DROP POLICY IF EXISTS "Users can upload own avatar" ON storage.objects;
CREATE POLICY "Users can upload own avatar"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'avatars'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can update own avatar" ON storage.objects;
CREATE POLICY "Users can update own avatar"
    ON storage.objects FOR UPDATE
    USING (
        bucket_id = 'avatars'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can delete own avatar" ON storage.objects;
CREATE POLICY "Users can delete own avatar"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'avatars'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
