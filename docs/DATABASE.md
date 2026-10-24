# StudySnap AI — Database Architecture & Supabase Integration Guide

## 1. Unified Architecture: Supabase Only

StudySnap AI uses **Supabase** as the single primary application database and authentication provider, paired with **ChromaDB** strictly for local vector embeddings.

The planned MongoDB integration has been completely removed. No MongoDB server, MongoDB API keys, or `MONGODB_URI` environment variables are required.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SUPABASE AUTH (Identity Provider)                     │
│  - User Registration, Login, Logout, Session Refresh, JWT Tokens            │
│  - Password Recovery via Email (`resetPasswordForEmail`)                    │
│  - Provides verified User UUID ('sub') embedded in Bearer tokens            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼ (Authorization: Bearer <JWT>)
┌─────────────────────────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND & AUTHORIZATION LAYER                      │
│  - auth.py: Decodes and verifies Supabase JWT token                          │
│  - Enforces User Ownership: record.owner_id === current_user.id             │
│  - postgrest.auth(access_token) propagates JWT for PostgreSQL RLS           │
└───────────────────────┬───────────────────────────────┬─────────────────────┘
                        │                               │
                        ▼                               ▼
         ┌─────────────────────────────┐  ┌───────────────────────────┐
         │     SUPABASE POSTGRESQL     │  │    CHROMADB VECTOR DB     │
         │   (Primary Application DB)  │  │   (backend/chroma_data/)  │
         │  - profiles                 │  │  - studysnap_collection   │
         │  - notebooks                │  │  - Source embeddings      │
         │  - documents                │  │  - Chunks & Page Numbers  │
         │  - notes                    │  │  - Cosine similarity      │
         │  - chat_sessions            │  └───────────────────────────┘
         │  - chat_messages            │
         │  - generated_materials      │
         │  - Row Level Security (RLS) │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │      SUPABASE STORAGE       │
         │   (Bucket: studysnap-assets)│
         │  - Lecture audio & notes    │
         │  - Generated infographics   │
         │  - Private RLS path scoping │
         └─────────────────────────────┘
```

---

## 2. Supabase PostgreSQL Database Schema

The complete version-controlled SQL migration is located at:
[`supabase/migrations/20261017000001_initial_studysnap_schema.sql`](file:///d:/mega_project/supabase/migrations/20261017000001_initial_studysnap_schema.sql)

### 2.1. `profiles`
User metadata extending Supabase Auth (`auth.users`):
- `id` (UUID, Primary Key, references `auth.users(id)` ON DELETE CASCADE)
- `display_name` (TEXT)
- `avatar_url` (TEXT)
- `college` (TEXT, default 'DKTE')
- `created_at` (TIMESTAMPTZ, default `timezone('utc', now())`)
- `updated_at` (TIMESTAMPTZ, default `timezone('utc', now())`)

### 2.2. `notebooks`
Primary workspace unit for each student:
- `id` (UUID, Primary Key, default `gen_random_uuid()`)
- `owner_id` (UUID, references `auth.users(id)` ON DELETE CASCADE)
- `title` (TEXT, NOT NULL, default 'Untitled Notebook')
- `description` (TEXT)
- `code` (TEXT, default 'CS101')
- `content` (TEXT, default '')
- `sources_count` (INTEGER, default 0)
- `notes_count` (INTEGER, default 0)
- `status` (TEXT, default 'ready')
- `created_at` (TIMESTAMPTZ, default `timezone('utc', now())`)
- `updated_at` (TIMESTAMPTZ, default `timezone('utc', now())`)

**Indexes:**
- `CREATE INDEX idx_notebooks_owner_updated ON public.notebooks(owner_id, updated_at DESC);`

### 2.3. `documents`
Uploaded lecture source files and metadata:
- `id` (UUID, Primary Key, default `gen_random_uuid()`)
- `owner_id` (UUID, references `auth.users(id)` ON DELETE CASCADE)
- `notebook_id` (UUID, references `public.notebooks(id)` ON DELETE CASCADE)
- `filename` (TEXT, NOT NULL)
- `file_type` (TEXT)
- `file_size` (BIGINT, default 0)
- `storage_path` (TEXT)
- `processing_status` (TEXT, default 'completed')
- `chunk_count` (INTEGER, default 0)
- `metadata` (JSONB, default `'{}'::jsonb`)
- `created_at` (TIMESTAMPTZ, default `timezone('utc', now())`)
- `updated_at` (TIMESTAMPTZ, default `timezone('utc', now())`)

**Indexes:**
- `CREATE INDEX idx_documents_notebook ON public.documents(notebook_id);`
- `CREATE INDEX idx_documents_owner ON public.documents(owner_id);`

### 2.4. `notes`
Saved academic notes, summaries, and lecture transcripts:
- `id` (UUID, Primary Key, default `gen_random_uuid()`)
- `owner_id` (UUID, references `auth.users(id)` ON DELETE CASCADE)
- `notebook_id` (UUID, references `public.notebooks(id)` ON DELETE CASCADE)
- `title` (TEXT, NOT NULL, default 'Untitled Note')
- `subject` (TEXT)
- `chapter` (TEXT)
- `content` (TEXT, NOT NULL, default '')
- `summary` (TEXT)
- `keywords` (TEXT[], default `'{}'`)
- `created_at` (TIMESTAMPTZ, default `timezone('utc', now())`)
- `updated_at` (TIMESTAMPTZ, default `timezone('utc', now())`)

**Indexes:**
- `CREATE INDEX idx_notes_notebook ON public.notes(notebook_id);`
- `CREATE INDEX idx_notes_owner ON public.notes(owner_id);`

### 2.5. `chat_sessions` & `chat_messages`
RAG conversational sessions and individual turn history:
- `chat_sessions`: `id`, `owner_id`, `notebook_id`, `title`, `selected_source_ids`, `created_at`, `updated_at`
- `chat_messages`: `id`, `session_id`, `owner_id`, `role` (user/assistant/system), `message_content`, `source_references` (JSONB), `creation_time`

**Indexes:**
- `CREATE INDEX idx_chat_sessions_notebook ON public.chat_sessions(notebook_id);`
- `CREATE INDEX idx_chat_messages_session ON public.chat_messages(session_id, creation_time ASC);`

### 2.6. `generated_materials`
AI-generated study artifacts (Audio Overviews, Infographics, Quizzes, Flashcards, Mind Maps):
- `id` (UUID, Primary Key, default `gen_random_uuid()`)
- `owner_id` (UUID, references `auth.users(id)` ON DELETE CASCADE)
- `notebook_id` (UUID, references `public.notebooks(id)` ON DELETE CASCADE)
- `material_type` (TEXT, NOT NULL)
- `title` (TEXT, NOT NULL)
- `storage_path` (TEXT)
- `metadata` (JSONB, default `'{}'::jsonb`)
- `created_at` (TIMESTAMPTZ, default `timezone('utc', now())`)
- `updated_at` (TIMESTAMPTZ, default `timezone('utc', now())`)

---

## 3. Row Level Security (RLS) & Access Control

Row Level Security is enabled on **every** table:
```sql
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notebooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.generated_materials ENABLE ROW LEVEL SECURITY;
```

Each table is governed by four standard policies:
- **SELECT**: `auth.uid() = owner_id` (or `auth.uid() = id` for `profiles`)
- **INSERT**: `WITH CHECK (auth.uid() = owner_id)`
- **UPDATE**: `USING (auth.uid() = owner_id) WITH CHECK (auth.uid() = owner_id)`
- **DELETE**: `USING (auth.uid() = owner_id)`

This guarantees that:
1. No user can read, query, update, or delete another student's notebooks or study materials.
2. Even direct PostgREST client queries from the frontend cannot leak cross-user records.
3. The FastAPI backend validates tokens with `auth.py` and passes user tokens to PostgREST to preserve RLS guarantees.

---

## 4. Supabase Storage Policies

A private storage bucket named `studysnap-assets` stores uploaded source PDFs, audio recordings, and generated assets:
```sql
-- Storage path convention: <owner_id>/<file_name>
CREATE POLICY "Users can access own storage files"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'studysnap-assets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can upload own storage files"
    ON storage.objects FOR INSERT
    WITH CHECK (bucket_id = 'studysnap-assets' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete own storage files"
    ON storage.objects FOR DELETE
    USING (bucket_id = 'studysnap-assets' AND auth.uid()::text = (storage.foldername(name))[1]);
```

---

## 5. Applying the Database Migration

### Method A: Supabase Dashboard (Recommended for Cloud)
1. Go to your [Supabase Dashboard](https://supabase.com/dashboard).
2. Open your project and navigate to the **SQL Editor** tab.
3. Open `supabase/migrations/20261017000001_initial_studysnap_schema.sql`.
4. Copy the entire file content, paste it into the editor, and click **Run**.
5. All 7 tables, indexes, automatic triggers, and RLS policies are created instantaneously.

### Method B: Supabase CLI
```bash
# Link project
supabase link --project-ref your-project-id

# Push migrations
supabase db push
```

---

## 6. Environment Configuration

### Frontend (`frontend/.env`)
```bash
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
VITE_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
VITE_AI_BACKEND_URL=http://localhost:8000
```

### Backend (`backend/.env`)
```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
SUPABASE_SECRET_KEY=your-supabase-service-role-key
SUPABASE_JWT_SECRET=your-supabase-jwt-secret
SUPABASE_STORAGE_BUCKET=studysnap-assets
LM_STUDIO_URL=http://localhost:1234/v1
```
