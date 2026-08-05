-- ── Anonymous workspaces: tenancy without accounts ──
-- Migration: 20260807000000_anonymous_workspaces.sql
--
-- TruthLens is open to anyone: there is no sign-up, no email, no password. Tenancy is carried by
-- a high-entropy workspace token the browser mints on first use and stores locally. The server
-- only ever sees the token's SHA-256 hash, so a database leak does not hand out workspace access.
--
-- SECURITY MODEL — stated plainly, because it is a real trade-off:
--   * The token is a bearer secret, like an unlisted share link. Anyone holding it can read and
--     write that workspace. There is no second factor and no account recovery.
--   * Losing the token means losing the workspace. Nothing can restore it, by design — we hold
--     no identifier that could prove ownership.
--   * This is appropriate for open, self-serve use. It is NOT sufficient for regulated personal
--     or health data; that needs real authentication, which this migration does not provide.
--
-- Because there is no auth.uid(), RLS cannot express "my rows". Instead every table is closed to
-- the anon and authenticated roles entirely, and all access goes through the API, which holds the
-- service-role key and scopes every query by workspace. RLS here is defence in depth: if a client
-- ever reached PostgREST directly with an anon key, it would see nothing.

/* ── Workspaces ─────────────────────────────────────────────────────────── */

ALTER TABLE public.organizations
    ADD COLUMN IF NOT EXISTS token_hash TEXT,
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS document_count INT NOT NULL DEFAULT 0;

-- Deliberately NOT a partial index. PostgREST's upsert emits a bare ON CONFLICT (token_hash),
-- and PostgreSQL will only use a partial unique index as the arbiter when the statement repeats
-- its predicate — which PostgREST cannot do. A plain unique index still permits many NULL rows,
-- so pre-existing organizations without a token are unaffected.
CREATE UNIQUE INDEX IF NOT EXISTS organizations_token_hash_idx ON public.organizations(token_hash);
CREATE INDEX IF NOT EXISTS organizations_last_seen_idx ON public.organizations(last_seen_at);

/* ── Reviewers are self-declared display names, not accounts ─────────────── */

ALTER TABLE public.review_decisions
    ADD COLUMN IF NOT EXISTS reviewer_name TEXT NOT NULL DEFAULT 'Anonymous reviewer';

ALTER TABLE public.review_tasks
    ADD COLUMN IF NOT EXISTS assigned_name TEXT;

-- These referenced auth.users, which no longer participates in the model.
ALTER TABLE public.review_decisions DROP CONSTRAINT IF EXISTS review_decisions_reviewer_id_fkey;
ALTER TABLE public.review_tasks DROP CONSTRAINT IF EXISTS review_tasks_assigned_to_fkey;
ALTER TABLE public.documents DROP CONSTRAINT IF EXISTS documents_created_by_fkey;
ALTER TABLE public.organization_members DROP CONSTRAINT IF EXISTS organization_members_user_id_fkey;

/* ── Batch jobs ──────────────────────────────────────────────────────────── */

ALTER TABLE public.verification_jobs
    ADD COLUMN IF NOT EXISTS label TEXT,
    ADD COLUMN IF NOT EXISTS error_detail TEXT;

CREATE TABLE IF NOT EXISTS public.verification_job_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID NOT NULL REFERENCES public.verification_jobs(id) ON DELETE CASCADE,
    document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL,
    file_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    trust_score INT CHECK (trust_score BETWEEN 0 AND 100),
    total_claims INT NOT NULL DEFAULT 0,
    needs_review_claims INT NOT NULL DEFAULT 0,
    error_detail TEXT,
    position INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS job_items_job_idx ON public.verification_job_items(job_id, position);

ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES public.verification_jobs(id) ON DELETE SET NULL;

/* ── Benchmark runs ──────────────────────────────────────────────────────── */

ALTER TABLE public.model_benchmarks
    ADD COLUMN IF NOT EXISTS run_id UUID,
    ADD COLUMN IF NOT EXISTS model_label TEXT,
    ADD COLUMN IF NOT EXISTS verified_claims INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS needs_review_claims INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS signals_measured_avg NUMERIC(3,1),
    ADD COLUMN IF NOT EXISTS evidence_cited INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS evidence_retrieved INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS failed BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS error_detail TEXT;

CREATE INDEX IF NOT EXISTS model_benchmarks_run_idx ON public.model_benchmarks(run_id, provider_id);

/* ── API activity log, for the compliance surface ────────────────────────── */

CREATE TABLE IF NOT EXISTS public.api_activity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE,
    route TEXT NOT NULL,
    action TEXT NOT NULL,
    status_code INT NOT NULL,
    detail TEXT,
    duration_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_activity_org_idx ON public.api_activity(organization_id, created_at DESC);

/* ── Denormalised workspace counter ──────────────────────────────────────── */

CREATE OR REPLACE FUNCTION public.increment_document_count(workspace UUID)
RETURNS VOID
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.organizations
    SET document_count = document_count + 1, last_seen_at = now()
    WHERE id = workspace;
$$;

/* ── Retention ───────────────────────────────────────────────────────────── */

-- Re-stamp expiry on documents already stored when a workspace changes its policy. Without this,
-- shortening retention would only ever apply to future uploads, which is not what a user means.
CREATE OR REPLACE FUNCTION public.reapply_retention(workspace UUID, days INT)
RETURNS VOID
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.documents
    SET retention_until = created_at + make_interval(days => days)
    WHERE organization_id = workspace;
$$;



CREATE OR REPLACE FUNCTION public.purge_expired_documents()
RETURNS INT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    removed INT;
BEGIN
    -- Claims, evidence, relations, timeline, audit rows and review records all cascade from here.
    WITH deleted AS (
        DELETE FROM public.documents
        WHERE retention_until IS NOT NULL AND retention_until < now()
        RETURNING 1
    )
    SELECT count(*) INTO removed FROM deleted;
    RETURN removed;
END;
$$;

/* ── Lock every table to the service role ────────────────────────────────── */

DO $$
DECLARE
    target TEXT;
BEGIN
    FOREACH target IN ARRAY ARRAY[
        'documents', 'claims', 'claim_evidence', 'claim_relations', 'verification_timeline',
        'audit_trails', 'organizations', 'organization_members', 'review_tasks',
        'review_decisions', 'verification_jobs', 'verification_job_items', 'model_benchmarks',
        'api_activity'
    ]
    LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', target);
        EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', target);
    END LOOP;
END $$;

-- The auth.uid()-based policies from earlier migrations can never match now that nothing
-- authenticates. Drop them so the deny-by-default posture is unambiguous rather than looking
-- like access control that happens to be failing.
DO $$
DECLARE
    policy RECORD;
BEGIN
    FOR policy IN
        SELECT policyname, tablename FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename IN (
            'documents', 'claims', 'claim_evidence', 'claim_relations', 'verification_timeline',
            'audit_trails', 'organizations', 'organization_members', 'review_tasks',
            'review_decisions', 'verification_jobs', 'model_benchmarks'
          )
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', policy.policyname, policy.tablename);
    END LOOP;
END $$;

DROP FUNCTION IF EXISTS public.is_org_member(UUID);
DROP FUNCTION IF EXISTS public.has_org_role(UUID, TEXT[]);
