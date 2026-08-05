-- ── Authenticated users replace anonymous workspaces ──
-- Migration: 20260808000000_authenticated_users.sql
--
-- The anonymous workspace token made tenancy depend on a bearer secret in localStorage: losing it
-- lost the data, sharing it shared everything, and the audit trail could only record a name the
-- reviewer typed themselves. An audit trail whose reviewer field is self-declared is not an audit
-- trail. Tenancy is now the Supabase auth user id, and reviewer identity comes from the verified
-- session.
--
-- Additive and idempotent: existing workspace rows are left in place, so an application rollback
-- is safe and no data is destroyed.

/* ── Tenancy key on every owned table ──────────────────────────────────── */

ALTER TABLE public.documents      ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE public.verification_jobs ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE public.model_benchmarks  ADD COLUMN IF NOT EXISTS user_id UUID;
ALTER TABLE public.api_activity      ADD COLUMN IF NOT EXISTS user_id UUID;

CREATE INDEX IF NOT EXISTS documents_user_idx        ON public.documents(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS api_activity_user_idx     ON public.api_activity(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS model_benchmarks_user_idx ON public.model_benchmarks(user_id, created_at DESC);

/* ── Reviewer identity comes from the session, not from a text box ─────── */

ALTER TABLE public.review_decisions
    ADD COLUMN IF NOT EXISTS reviewer_user_id UUID,
    ADD COLUMN IF NOT EXISTS reviewer_email TEXT;

ALTER TABLE public.review_tasks
    ADD COLUMN IF NOT EXISTS assigned_user_id UUID,
    ADD COLUMN IF NOT EXISTS assigned_email TEXT;

ALTER TABLE public.audit_trails
    ADD COLUMN IF NOT EXISTS reviewer_user_id UUID,
    ADD COLUMN IF NOT EXISTS reviewer_email TEXT;

/* ── Per-user preferences, so a signed-in user carries settings across devices ── */

CREATE TABLE IF NOT EXISTS public.user_settings (
    user_id UUID PRIMARY KEY,
    display_name TEXT,
    email TEXT,
    retention_days INT NOT NULL DEFAULT 30 CHECK (retention_days BETWEEN 1 AND 3650),
    preferred_provider TEXT,
    preferred_model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

/* ── Retention, re-expressed per user ──────────────────────────────────── */

CREATE OR REPLACE FUNCTION public.reapply_retention_for_user(target_user UUID, days INT)
RETURNS VOID
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.documents
    SET retention_until = created_at + make_interval(days => days)
    WHERE user_id = target_user;
$$;

/* ── Lock everything to the service role ───────────────────────────────── */
-- Nothing authenticates directly against PostgREST: the browser talks only to /api, which holds
-- the service-role key and scopes every query by the verified user id. RLS stays enabled with no
-- client-facing policy, so a leaked anon key still yields nothing.

DO $$
DECLARE
    target TEXT;
BEGIN
    FOREACH target IN ARRAY ARRAY['user_settings']
    LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', target);
        EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', target);
    END LOOP;
END $$;
