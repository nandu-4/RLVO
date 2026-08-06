-- TruthLens v1: Google/Supabase Auth users, guest demo mode, workspace, review, audit and replay.
-- Guests never write database rows. The browser is denied direct table access; the API service role
-- verifies each Supabase session and scopes every operation by user_id.

CREATE TABLE public.documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  document_name TEXT NOT NULL, document_type TEXT NOT NULL DEFAULT 'Enterprise Document', file_size_kb INT NOT NULL DEFAULT 0,
  provider TEXT NOT NULL, model TEXT NOT NULL, verification_mode TEXT NOT NULL CHECK (verification_mode IN ('cross_check','self_check')),
  trust_score INT NOT NULL DEFAULT 0 CHECK (trust_score BETWEEN 0 AND 100), risk_level TEXT NOT NULL DEFAULT 'LOW',
  total_claims INT NOT NULL DEFAULT 0, verified_claims INT NOT NULL DEFAULT 0, corrected_claims INT NOT NULL DEFAULT 0, unsupported_claims INT NOT NULL DEFAULT 0, needs_review_claims INT NOT NULL DEFAULT 0,
  retention_until TIMESTAMPTZ, processing_status TEXT NOT NULL DEFAULT 'completed' CHECK (processing_status IN ('queued','processing','completed','failed')),
  mean_legibility INT CHECK (mean_legibility BETWEEN 0 AND 100), indexed_blocks INT NOT NULL DEFAULT 0, page_count INT NOT NULL DEFAULT 1,
  ocr_engine TEXT, provider_latency_ms INT, document_processing_time_ms INT, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.claims (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  field_name TEXT NOT NULL, category TEXT, original_value TEXT NOT NULL, verified_value TEXT, status TEXT NOT NULL CHECK (status IN ('verified','corrected','unsupported','needs_review')),
  trust_score INT NOT NULL DEFAULT 0 CHECK (trust_score BETWEEN 0 AND 100), reason TEXT NOT NULL,
  ocr_agreement INT DEFAULT 0, vision_agreement INT DEFAULT 0, layout_agreement INT DEFAULT 0, semantic_agreement INT DEFAULT 0, evidence_strength INT DEFAULT 0,
  signals_measured INT NOT NULL DEFAULT 0 CHECK (signals_measured BETWEEN 0 AND 5), hallucination_risk_level TEXT CHECK (hallucination_risk_level IN ('LOW','MEDIUM','HIGH')),
  hallucination_risk_score INT CHECK (hallucination_risk_score BETWEEN 0 AND 100), score_rationale JSONB, retrieval_candidates INT NOT NULL DEFAULT 0, retrieval_cited INT NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.claim_evidence (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
  evidence_type TEXT NOT NULL, source_name TEXT NOT NULL, extracted_text TEXT NOT NULL, page_number INT NOT NULL DEFAULT 1,
  bounding_box_json JSONB, layout_region TEXT, confidence INT NOT NULL DEFAULT 0, retrieved_by TEXT[] NOT NULL DEFAULT '{}', cited BOOLEAN NOT NULL DEFAULT TRUE, image_crop_url TEXT, ocr_block_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE public.claim_relations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  from_claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE, to_claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
  kind TEXT NOT NULL, strength NUMERIC(3,2) NOT NULL DEFAULT 0 CHECK (strength BETWEEN 0 AND 1), created_at TIMESTAMPTZ NOT NULL DEFAULT now(), CHECK (from_claim_id <> to_claim_id)
);
CREATE TABLE public.verification_timeline (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE, step_name TEXT NOT NULL, event_title TEXT NOT NULL, event_detail TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'info', timestamp_formatted TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.audit_trails (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE, claim_id UUID REFERENCES public.claims(id) ON DELETE CASCADE, file_name TEXT NOT NULL, field_name TEXT NOT NULL, original_value TEXT NOT NULL, final_value TEXT NOT NULL, status TEXT NOT NULL, trust_score INT NOT NULL, reviewer_user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, reviewer_notes TEXT, timestamp TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.review_tasks (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE, claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE, status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','assigned','resolved')), assigned_user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), resolved_at TIMESTAMPTZ);
CREATE TABLE public.review_decisions (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE, claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE, decision TEXT NOT NULL CHECK (decision IN ('approved','rejected','overridden')), reviewer_user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL, reviewer_notes TEXT, override_value TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.model_benchmarks (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE, run_id UUID, document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL, provider_id TEXT NOT NULL, model_label TEXT, claims_generated INT NOT NULL DEFAULT 0, verified_claims INT NOT NULL DEFAULT 0, corrections INT NOT NULL DEFAULT 0, unsupported_claims INT NOT NULL DEFAULT 0, needs_review_claims INT NOT NULL DEFAULT 0, trust_score INT, accuracy NUMERIC(5,2), signals_measured_avg NUMERIC(5,2), evidence_cited INT NOT NULL DEFAULT 0, verification_time_ms INT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.activity_logs (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE, route TEXT NOT NULL, action TEXT NOT NULL, status_code INT NOT NULL, detail TEXT, duration_ms INT, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.profiles (id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE, display_name TEXT, avatar_url TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.user_settings (user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE, display_name TEXT, email TEXT, retention_days INT NOT NULL DEFAULT 30 CHECK (retention_days BETWEEN 1 AND 3650), preferred_provider TEXT, preferred_model TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE public.verification_replays (document_id UUID PRIMARY KEY REFERENCES public.documents(id) ON DELETE CASCADE, user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE, verification_snapshot JSONB NOT NULL, provider_attempts JSONB NOT NULL DEFAULT '[]'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT now());

CREATE INDEX documents_user_created_idx ON public.documents(user_id, created_at DESC); CREATE INDEX documents_retention_idx ON public.documents(retention_until); CREATE INDEX claims_document_idx ON public.claims(document_id); CREATE INDEX evidence_claim_idx ON public.claim_evidence(claim_id); CREATE INDEX replay_user_created_idx ON public.verification_replays(user_id, created_at DESC); CREATE INDEX activity_user_created_idx ON public.activity_logs(user_id, created_at DESC);

CREATE OR REPLACE FUNCTION public.reapply_retention_for_user(target_user UUID, days INT) RETURNS VOID LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$ UPDATE public.documents SET retention_until = created_at + make_interval(days => days) WHERE user_id = target_user; $$;

-- Service-role-only persistence: RLS remains enabled as a defence in depth for the public anon key.
DO $$ DECLARE t TEXT; BEGIN FOREACH t IN ARRAY ARRAY['documents','claims','claim_evidence','claim_relations','verification_timeline','audit_trails','review_tasks','review_decisions','model_benchmarks','activity_logs','profiles','user_settings','verification_replays'] LOOP EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t); EXECUTE format('REVOKE ALL ON public.%I FROM anon, authenticated', t); END LOOP; END $$;
