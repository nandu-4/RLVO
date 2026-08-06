-- Restore service_role privileges.
--
-- WHY THIS IS NEEDED
-- 20260801000000_truthlens_v1.sql enables RLS on every table and revokes all privileges from
-- anon and authenticated, but never grants anything to service_role. On a project where the
-- public-schema default privileges do not auto-grant, that leaves NO role able to touch the
-- tables: the API authenticates as service_role and every request fails with
--
--   42501  permission denied for table documents
--   hint:  GRANT SELECT ON public.documents TO service_role;
--
-- The tables are present and correct — this is purely an access-control gap, and it locks out
-- the one role the application actually uses. Symptomatically it is indistinguishable from
-- "database not configured", which is what makes it worth stating plainly here.
--
-- The revoke from anon and authenticated is deliberate and is NOT undone. Every read and write
-- goes through the service role behind verified-identity checks in api/_identity.ts, so the
-- public keys are meant to have no direct table access. That is defence in depth, and it only
-- works if service_role itself is granted what it needs.
--
-- Idempotent: GRANT is a no-op when the privilege is already held.

DO $$
DECLARE
  t TEXT;
BEGIN
  FOREACH t IN ARRAY ARRAY[
    'documents', 'claims', 'claim_evidence', 'claim_relations', 'verification_timeline',
    'audit_trails', 'review_tasks', 'review_decisions', 'model_benchmarks', 'activity_logs',
    'profiles', 'user_settings', 'verification_replays'
  ]
  LOOP
    -- to_regclass returns NULL rather than raising, so a table this migration does not know
    -- about cannot abort the whole run.
    IF to_regclass('public.' || quote_ident(t)) IS NOT NULL THEN
      EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON public.%I TO service_role', t);
      EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', t);
    END IF;
  END LOOP;
END $$;

-- Schema-level access, without which the grants above are unreachable.
GRANT USAGE ON SCHEMA public TO service_role;

-- Identity/serial columns, if any table ever gains one.
DO $$
DECLARE
  s TEXT;
BEGIN
  FOR s IN SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public'
  LOOP
    EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE public.%I TO service_role', s);
  END LOOP;
END $$;

-- SECURITY DEFINER helpers the API calls by name.
DO $$
DECLARE
  f RECORD;
BEGIN
  FOR f IN
    SELECT p.oid::regprocedure AS sig
    FROM pg_proc p
    JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE n.nspname = 'public'
  LOOP
    EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', f.sig);
  END LOOP;
END $$;

-- Anything added later inherits the same grants, so this gap cannot silently return.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO service_role;
