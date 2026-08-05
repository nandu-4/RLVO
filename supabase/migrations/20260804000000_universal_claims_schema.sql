-- ── TruthLens Enterprise AI Hallucination Verification Platform ──
-- Migration: 20260804000000_universal_claims_schema.sql

CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_name TEXT NOT NULL,
    document_type TEXT NOT NULL DEFAULT 'Enterprise Document',
    file_size_kb INT DEFAULT 0,
    file_url TEXT,
    model_used TEXT DEFAULT 'gemini-2.5',
    trust_score INT DEFAULT 0,
    risk_level TEXT DEFAULT 'LOW',
    total_claims INT DEFAULT 0,
    verified_claims INT DEFAULT 0,
    corrected_claims INT DEFAULT 0,
    unsupported_claims INT DEFAULT 0,
    needs_review_claims INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    category TEXT,
    original_value TEXT NOT NULL,
    verified_value TEXT,
    status TEXT NOT NULL CHECK (status IN ('verified', 'corrected', 'unsupported', 'needs_review')),
    trust_score INT NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    ocr_agreement INT DEFAULT 0,
    vision_agreement INT DEFAULT 0,
    layout_agreement INT DEFAULT 0,
    semantic_agreement INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.claim_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id UUID REFERENCES public.claims(id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL CHECK (evidence_type IN ('ocr', 'vision', 'layout', 'retrieval', 'metadata')),
    source_name TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    page_number INT DEFAULT 1,
    bounding_box_json JSONB,
    layout_region TEXT,
    confidence INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.verification_timeline (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    event_title TEXT NOT NULL,
    event_detail TEXT NOT NULL,
    status TEXT DEFAULT 'info',
    timestamp_formatted TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.audit_trails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE,
    claim_id UUID REFERENCES public.claims(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    field_name TEXT NOT NULL,
    original_value TEXT NOT NULL,
    final_value TEXT NOT NULL,
    status TEXT NOT NULL,
    trust_score INT NOT NULL,
    reviewer_name TEXT NOT NULL DEFAULT 'Automated Engine',
    reviewer_notes TEXT,
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- Enterprise tenancy, human review and asynchronous-processing foundations.
CREATE TABLE IF NOT EXISTS public.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    retention_days INT NOT NULL DEFAULT 30 CHECK (retention_days BETWEEN 1 AND 3650),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.organization_members (
    organization_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'reviewer', 'member', 'viewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (organization_id, user_id)
);

ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE;
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS created_by UUID REFERENCES auth.users(id);
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ;
ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS processing_status TEXT NOT NULL DEFAULT 'completed' CHECK (processing_status IN ('queued', 'processing', 'completed', 'failed'));

CREATE TABLE IF NOT EXISTS public.review_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
    assigned_to UUID REFERENCES auth.users(id),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'assigned', 'resolved')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS public.review_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
    reviewer_id UUID REFERENCES auth.users(id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'rejected', 'overridden')),
    reviewer_notes TEXT,
    override_value TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.verification_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    total_documents INT NOT NULL DEFAULT 0,
    completed_documents INT NOT NULL DEFAULT 0,
    failed_documents INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.model_benchmarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES public.organizations(id) ON DELETE CASCADE,
    document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL,
    provider_id TEXT NOT NULL,
    claims_generated INT NOT NULL DEFAULT 0,
    corrections INT NOT NULL DEFAULT 0,
    unsupported_claims INT NOT NULL DEFAULT 0,
    trust_score INT CHECK (trust_score BETWEEN 0 AND 100),
    accuracy NUMERIC(5,2),
    verification_time_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_organization_created_at_idx ON public.documents(organization_id, created_at DESC);
CREATE INDEX IF NOT EXISTS review_tasks_assignee_status_idx ON public.review_tasks(assigned_to, status);
CREATE INDEX IF NOT EXISTS claims_document_status_idx ON public.claims(document_id, status);

-- RLS Policies
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.claim_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verification_timeline ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_trails ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Allow public read access to documents" ON public.documents;
DROP POLICY IF EXISTS "Allow public insert to documents" ON public.documents;
DROP POLICY IF EXISTS "Organization members can read documents" ON public.documents;
CREATE POLICY "Organization members can read documents" ON public.documents FOR SELECT USING (organization_id IN (SELECT organization_id FROM public.organization_members WHERE user_id = auth.uid()));
DROP POLICY IF EXISTS "Organization members can insert documents" ON public.documents;
CREATE POLICY "Organization members can insert documents" ON public.documents FOR INSERT WITH CHECK (organization_id IN (SELECT organization_id FROM public.organization_members WHERE user_id = auth.uid()));

DROP POLICY IF EXISTS "Allow public read access to claims" ON public.claims;
DROP POLICY IF EXISTS "Allow public insert to claims" ON public.claims;
DROP POLICY IF EXISTS "Organization members can read claims" ON public.claims;
CREATE POLICY "Organization members can read claims" ON public.claims FOR SELECT USING (EXISTS (SELECT 1 FROM public.documents d JOIN public.organization_members m ON m.organization_id = d.organization_id WHERE d.id = claims.document_id AND m.user_id = auth.uid()));

DROP POLICY IF EXISTS "Allow public read access to evidence" ON public.claim_evidence;
DROP POLICY IF EXISTS "Allow public insert to evidence" ON public.claim_evidence;

DROP POLICY IF EXISTS "Allow public read access to timeline" ON public.verification_timeline;
DROP POLICY IF EXISTS "Allow public insert to timeline" ON public.verification_timeline;

DROP POLICY IF EXISTS "Allow public read access to audit_trails" ON public.audit_trails;
DROP POLICY IF EXISTS "Allow public insert to audit_trails" ON public.audit_trails;

ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.review_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.review_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.verification_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_benchmarks ENABLE ROW LEVEL SECURITY;
-- Every CREATE POLICY is preceded by a DROP so the migration can be re-applied without error.
DROP POLICY IF EXISTS "Members can view their organizations" ON public.organizations;
CREATE POLICY "Members can view their organizations" ON public.organizations FOR SELECT USING (id IN (SELECT organization_id FROM public.organization_members WHERE user_id = auth.uid()));
DROP POLICY IF EXISTS "Members can view organization membership" ON public.organization_members;
CREATE POLICY "Members can view organization membership" ON public.organization_members FOR SELECT USING (organization_id IN (SELECT organization_id FROM public.organization_members WHERE user_id = auth.uid()));
DROP POLICY IF EXISTS "Reviewers can view review tasks" ON public.review_tasks;
CREATE POLICY "Reviewers can view review tasks" ON public.review_tasks FOR SELECT USING (assigned_to = auth.uid());
DROP POLICY IF EXISTS "Reviewers can create decisions for assigned tasks" ON public.review_decisions;
CREATE POLICY "Reviewers can create decisions for assigned tasks" ON public.review_decisions FOR INSERT WITH CHECK (reviewer_id = auth.uid());
DROP POLICY IF EXISTS "Members can view model benchmarks" ON public.model_benchmarks;
CREATE POLICY "Members can view model benchmarks" ON public.model_benchmarks FOR SELECT USING (organization_id IN (SELECT organization_id FROM public.organization_members WHERE user_id = auth.uid()));
