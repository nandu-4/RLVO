-- Restore the batch-job tables and documents.job_id.
--
-- WHY THIS IS NEEDED
-- api/_persistence.ts writes `job_id` on every documents insert — not just batch runs; a normal
-- single-document verification sends `job_id: null`. The consolidated migration
-- 20260801000000_truthlens_v1.sql dropped both job tables and that column, so PostgREST rejected
-- the whole insert:
--
--   PGRST204  Could not find the 'job_id' column of 'documents' in the schema cache
--
-- The knock-on effect is what made this hard to see from the UI. The documents insert failed, so
-- `documentId` stayed null; the replay snapshot was then written against a freshly generated UUID
-- that had no matching row, and failed its foreign key:
--
--   23503  Key (document_id)=(...) is not present in table "documents"
--
-- Verification itself still returned 200 and the timeline recorded the write failure, so the user
-- saw correct results and an empty History and Dashboard, with nothing obviously broken.
--
-- The column is restored rather than removed from the code because batch verification still
-- exists as an API surface (api/batch-job.ts) — it is only withdrawn from the navigation. Deleting
-- the write would have meant deleting that feature outright, which is a product decision, not a
-- schema fix.
--
-- Shape follows the CURRENT identity model: ownership is `user_id` referencing auth.users, the
-- same as every other table here. The pre-consolidation definition used `organization_id`, which
-- belonged to an organisations model this project no longer has.
--
-- Idempotent throughout.

CREATE TABLE IF NOT EXISTS public.verification_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    label TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    total_documents INT NOT NULL DEFAULT 0,
    completed_documents INT NOT NULL DEFAULT 0,
    failed_documents INT NOT NULL DEFAULT 0,
    error_detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.verification_job_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES public.verification_jobs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL,
    file_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    trust_score INT CHECK (trust_score BETWEEN 0 AND 100),
    total_claims INT NOT NULL DEFAULT 0,
    needs_review_claims INT NOT NULL DEFAULT 0,
    error_detail TEXT,
    position INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS job_items_job_idx ON public.verification_job_items(job_id, position);
CREATE INDEX IF NOT EXISTS jobs_user_created_idx ON public.verification_jobs(user_id, created_at DESC);

-- The column whose absence broke every single-document write.
ALTER TABLE public.documents
    ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.verification_jobs(id) ON DELETE SET NULL;

-- Same posture as every other table: RLS on, public roles revoked, service_role granted.
DO $$
DECLARE
  t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY['verification_jobs', 'verification_job_items']
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', t);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO service_role', t);
  END LOOP;
END $$;
