-- =============================================================================
-- StudySnap AI — Supabase PostgreSQL Schema & Row Level Security (RLS)
-- Migration: 20261017000001_initial_studysnap_schema.sql
-- Self-healing migration: 100% idempotent and compatible with both fresh
-- databases and pre-existing legacy tables with user_id or missing columns.
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- 1. BASE TABLES DEFINITION (CREATE IF NOT EXISTS)
-- =============================================================================

-- 1.1 User Profiles
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT,
    avatar_url TEXT,
    college TEXT DEFAULT 'DKTE',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 1.2 Notebooks (Study workspace units)
CREATE TABLE IF NOT EXISTS public.notebooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'Untitled Notebook',
    description TEXT DEFAULT '',
    code TEXT DEFAULT 'CS101',
    content TEXT DEFAULT '',
    sources_count INTEGER NOT NULL DEFAULT 0,
    notes_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 1.3 Documents (Source uploads & file metadata)
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    filename TEXT NOT NULL DEFAULT 'document',
    file_type TEXT,
    file_size BIGINT DEFAULT 0,
    storage_path TEXT,
    processing_status TEXT NOT NULL DEFAULT 'completed',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 1.4 Notes (Saved notes, summaries, lecture transcripts)
CREATE TABLE IF NOT EXISTS public.notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'Untitled Note',
    subject TEXT,
    chapter TEXT,
    content TEXT NOT NULL DEFAULT '',
    summary TEXT DEFAULT '',
    keywords TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 1.5 Chat Sessions & Messages (RAG conversational context)
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New Conversation',
    selected_source_ids TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE TABLE IF NOT EXISTS public.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'user',
    message_content TEXT NOT NULL DEFAULT '',
    source_references JSONB DEFAULT '[]'::jsonb,
    creation_time TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- 1.6 Generated Materials (Artifacts: Podcast Audio, Infographics, Quizzes, Flashcards)
CREATE TABLE IF NOT EXISTS public.generated_materials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    material_type TEXT NOT NULL DEFAULT 'general',
    title TEXT NOT NULL DEFAULT 'Study Material',
    storage_path TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now())
);

-- =============================================================================
-- 2. SCHEMA EVOLUTION & COLUMN REPAIRS (Runs unconditionally on pre-existing tables)
-- =============================================================================

-- 2.1 Notebooks columns
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'Untitled Notebook';
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS description TEXT DEFAULT '';
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS code TEXT DEFAULT 'CS101';
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS content TEXT DEFAULT '';
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS sources_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS notes_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'ready';
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE IF EXISTS public.notebooks ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.2 Documents columns
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS filename TEXT NOT NULL DEFAULT 'document';
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS file_type TEXT;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS file_size BIGINT DEFAULT 0;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS storage_path TEXT;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE IF EXISTS public.documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.3 Notes columns
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'Untitled Note';
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS subject TEXT;
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS chapter TEXT;
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS content TEXT NOT NULL DEFAULT '';
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT '';
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}';
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE IF EXISTS public.notes ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.4 Chat Sessions columns
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'New Conversation';
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS selected_source_ids TEXT[] DEFAULT '{}';
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE IF EXISTS public.chat_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.5 Chat Messages columns
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS session_id UUID REFERENCES public.chat_sessions(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user';
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS message_content TEXT NOT NULL DEFAULT '';
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS source_references JSONB DEFAULT '[]'::jsonb;
ALTER TABLE IF EXISTS public.chat_messages ADD COLUMN IF NOT EXISTS creation_time TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.6 Generated Materials columns
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE;
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS material_type TEXT NOT NULL DEFAULT 'general';
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS title TEXT NOT NULL DEFAULT 'Study Material';
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS storage_path TEXT;
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());
ALTER TABLE IF EXISTS public.generated_materials ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc'::text, now());

-- 2.7 Backfill owner_id <-> user_id across existing records
UPDATE public.notebooks SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.notebooks SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

UPDATE public.documents SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.documents SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

UPDATE public.notes SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.notes SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

UPDATE public.chat_sessions SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.chat_sessions SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

UPDATE public.chat_messages SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.chat_messages SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

-- Backfill chat_messages without owner_id from parent session
UPDATE public.chat_messages m
SET owner_id = s.owner_id, user_id = COALESCE(s.user_id, s.owner_id)
FROM public.chat_sessions s
WHERE m.session_id = s.id AND m.owner_id IS NULL;

UPDATE public.generated_materials SET owner_id = user_id WHERE owner_id IS NULL AND user_id IS NOT NULL;
UPDATE public.generated_materials SET user_id = owner_id WHERE user_id IS NULL AND owner_id IS NOT NULL;

-- =============================================================================
-- 3. BIDIRECTIONAL SYNC TRIGGERS (Keeps owner_id and user_id in sync automatically)
-- =============================================================================
CREATE OR REPLACE FUNCTION public.sync_owner_and_user_id()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.owner_id IS NULL AND NEW.user_id IS NOT NULL THEN
        NEW.owner_id = NEW.user_id;
    ELSIF NEW.user_id IS NULL AND NEW.owner_id IS NOT NULL THEN
        NEW.user_id = NEW.owner_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_sync_notebooks_owner ON public.notebooks;
CREATE TRIGGER trigger_sync_notebooks_owner BEFORE INSERT OR UPDATE ON public.notebooks
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

DROP TRIGGER IF EXISTS trigger_sync_documents_owner ON public.documents;
CREATE TRIGGER trigger_sync_documents_owner BEFORE INSERT OR UPDATE ON public.documents
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

DROP TRIGGER IF EXISTS trigger_sync_notes_owner ON public.notes;
CREATE TRIGGER trigger_sync_notes_owner BEFORE INSERT OR UPDATE ON public.notes
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

DROP TRIGGER IF EXISTS trigger_sync_chat_sessions_owner ON public.chat_sessions;
CREATE TRIGGER trigger_sync_chat_sessions_owner BEFORE INSERT OR UPDATE ON public.chat_sessions
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

DROP TRIGGER IF EXISTS trigger_sync_chat_messages_owner ON public.chat_messages;
CREATE TRIGGER trigger_sync_chat_messages_owner BEFORE INSERT OR UPDATE ON public.chat_messages
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

DROP TRIGGER IF EXISTS trigger_sync_generated_materials_owner ON public.generated_materials;
CREATE TRIGGER trigger_sync_generated_materials_owner BEFORE INSERT OR UPDATE ON public.generated_materials
FOR EACH ROW EXECUTE FUNCTION public.sync_owner_and_user_id();

-- =============================================================================
-- 4. PERFORMANCE INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_notebooks_owner_updated ON public.notebooks(owner_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_notebooks_user_updated ON public.notebooks(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_notebook ON public.documents(notebook_id);
CREATE INDEX IF NOT EXISTS idx_documents_owner ON public.documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_user ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_notes_notebook ON public.notes(notebook_id);
CREATE INDEX IF NOT EXISTS idx_notes_owner ON public.notes(owner_id);
CREATE INDEX IF NOT EXISTS idx_notes_user ON public.notes(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_notebook ON public.chat_sessions(notebook_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner ON public.chat_sessions(owner_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON public.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON public.chat_messages(session_id, creation_time ASC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_owner ON public.chat_messages(owner_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON public.chat_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_generated_materials_notebook ON public.generated_materials(notebook_id);
CREATE INDEX IF NOT EXISTS idx_generated_materials_owner ON public.generated_materials(owner_id);
CREATE INDEX IF NOT EXISTS idx_generated_materials_user ON public.generated_materials(user_id);

-- =============================================================================
-- 5. AUTOMATIC TIMESTAMP TRIGGERS
-- =============================================================================
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_profiles_updated_at ON public.profiles;
CREATE TRIGGER trigger_profiles_updated_at BEFORE UPDATE ON public.profiles
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS trigger_notebooks_updated_at ON public.notebooks;
CREATE TRIGGER trigger_notebooks_updated_at BEFORE UPDATE ON public.notebooks
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS trigger_documents_updated_at ON public.documents;
CREATE TRIGGER trigger_documents_updated_at BEFORE UPDATE ON public.documents
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS trigger_notes_updated_at ON public.notes;
CREATE TRIGGER trigger_notes_updated_at BEFORE UPDATE ON public.notes
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS trigger_chat_sessions_updated_at ON public.chat_sessions;
CREATE TRIGGER trigger_chat_sessions_updated_at BEFORE UPDATE ON public.chat_sessions
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

DROP TRIGGER IF EXISTS trigger_generated_materials_updated_at ON public.generated_materials;
CREATE TRIGGER trigger_generated_materials_updated_at BEFORE UPDATE ON public.generated_materials
FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Automatic profile creation on auth.users signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, display_name, college)
    VALUES (
        NEW.id,
        COALESCE(NEW.raw_user_meta_data->>'name', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'college', 'DKTE')
    )
    ON CONFLICT (id) DO UPDATE
    SET display_name = EXCLUDED.display_name,
        college = EXCLUDED.college,
        updated_at = timezone('utc'::text, now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =============================================================================
-- 6. ROW LEVEL SECURITY (RLS) POLICIES
-- =============================================================================

-- 6.1 Profiles RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
CREATE POLICY "Users can insert own profile"
    ON public.profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

-- 6.2 Notebooks RLS (Supports both owner_id and user_id)
ALTER TABLE public.notebooks ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own notebooks" ON public.notebooks;
CREATE POLICY "Users can select own notebooks"
    ON public.notebooks FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own notebooks" ON public.notebooks;
CREATE POLICY "Users can insert own notebooks"
    ON public.notebooks FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own notebooks" ON public.notebooks;
CREATE POLICY "Users can update own notebooks"
    ON public.notebooks FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own notebooks" ON public.notebooks;
CREATE POLICY "Users can delete own notebooks"
    ON public.notebooks FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- 6.3 Documents RLS
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own documents" ON public.documents;
CREATE POLICY "Users can select own documents"
    ON public.documents FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own documents" ON public.documents;
CREATE POLICY "Users can insert own documents"
    ON public.documents FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own documents" ON public.documents;
CREATE POLICY "Users can update own documents"
    ON public.documents FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own documents" ON public.documents;
CREATE POLICY "Users can delete own documents"
    ON public.documents FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- 6.4 Notes RLS
ALTER TABLE public.notes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own notes" ON public.notes;
CREATE POLICY "Users can select own notes"
    ON public.notes FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own notes" ON public.notes;
CREATE POLICY "Users can insert own notes"
    ON public.notes FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own notes" ON public.notes;
CREATE POLICY "Users can update own notes"
    ON public.notes FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own notes" ON public.notes;
CREATE POLICY "Users can delete own notes"
    ON public.notes FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- 6.5 Chat Sessions RLS
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own chat sessions" ON public.chat_sessions;
CREATE POLICY "Users can select own chat sessions"
    ON public.chat_sessions FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own chat sessions" ON public.chat_sessions;
CREATE POLICY "Users can insert own chat sessions"
    ON public.chat_sessions FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own chat sessions" ON public.chat_sessions;
CREATE POLICY "Users can update own chat sessions"
    ON public.chat_sessions FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own chat sessions" ON public.chat_sessions;
CREATE POLICY "Users can delete own chat sessions"
    ON public.chat_sessions FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- 6.6 Chat Messages RLS
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own chat messages" ON public.chat_messages;
CREATE POLICY "Users can select own chat messages"
    ON public.chat_messages FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own chat messages" ON public.chat_messages;
CREATE POLICY "Users can insert own chat messages"
    ON public.chat_messages FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own chat messages" ON public.chat_messages;
CREATE POLICY "Users can update own chat messages"
    ON public.chat_messages FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own chat messages" ON public.chat_messages;
CREATE POLICY "Users can delete own chat messages"
    ON public.chat_messages FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- 6.7 Generated Materials RLS
ALTER TABLE public.generated_materials ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can select own generated materials" ON public.generated_materials;
CREATE POLICY "Users can select own generated materials"
    ON public.generated_materials FOR SELECT
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can insert own generated materials" ON public.generated_materials;
CREATE POLICY "Users can insert own generated materials"
    ON public.generated_materials FOR INSERT
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can update own generated materials" ON public.generated_materials;
CREATE POLICY "Users can update own generated materials"
    ON public.generated_materials FOR UPDATE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id))
    WITH CHECK (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

DROP POLICY IF EXISTS "Users can delete own generated materials" ON public.generated_materials;
CREATE POLICY "Users can delete own generated materials"
    ON public.generated_materials FOR DELETE
    USING (auth.uid() = owner_id OR (owner_id IS NULL AND auth.uid() = user_id));

-- =============================================================================
-- 7. STORAGE BUCKET & STORAGE RLS POLICIES
-- =============================================================================

-- Ensure private storage bucket exists for studysnap-assets
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'studysnap-assets',
    'studysnap-assets',
    false,
    104857600, -- 100MB limit
    ARRAY['application/pdf', 'audio/mpeg', 'audio/ogg', 'audio/wav', 'image/png', 'image/jpeg', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/vnd.openxmlformats-officedocument.presentationml.presentation']
)
ON CONFLICT (id) DO NOTHING;

-- RLS policies for storage objects (user scoped by first folder segment: <user_id>/<file>)
DROP POLICY IF EXISTS "Users can access own storage files" ON storage.objects;
CREATE POLICY "Users can access own storage files"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'studysnap-assets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can upload own storage files" ON storage.objects;
CREATE POLICY "Users can upload own storage files"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'studysnap-assets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can update own storage files" ON storage.objects;
CREATE POLICY "Users can update own storage files"
    ON storage.objects FOR UPDATE
    USING (
        bucket_id = 'studysnap-assets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

DROP POLICY IF EXISTS "Users can delete own storage files" ON storage.objects;
CREATE POLICY "Users can delete own storage files"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'studysnap-assets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );
