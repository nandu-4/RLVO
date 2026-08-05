-- ── Fix RLS lockout and review-integrity gaps ──
-- Migration: 20260805000000_fix_rls_policy_lockout.sql
--
-- The previous migration enabled RLS on claim_evidence, verification_timeline and
-- audit_trails, DROPped their permissive policies, and never created replacements. With RLS
-- on and zero policies, PostgreSQL denies everything: those three tables became unreachable
-- for every non-service-role client. verification_jobs had the same problem. This restores
-- organization-scoped access and adds the write policies the review workflow needs.

-- Membership test as a stable function so every policy shares one definition.
CREATE OR REPLACE FUNCTION public.is_org_member(target_org UUID)
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.organization_members
        WHERE organization_id = target_org AND user_id = auth.uid()
    );
$$;

CREATE OR REPLACE FUNCTION public.has_org_role(target_org UUID, allowed TEXT[])
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT EXISTS (
        SELECT 1 FROM public.organization_members
        WHERE organization_id = target_org AND user_id = auth.uid() AND role = ANY(allowed)
    );
$$;

-- Resolve the owning organization for a claim, used by the child-table policies.
CREATE OR REPLACE FUNCTION public.claim_org(target_claim UUID)
RETURNS UUID
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT d.organization_id
    FROM public.claims c
    JOIN public.documents d ON d.id = c.document_id
    WHERE c.id = target_claim;
$$;

-- ── claim_evidence: previously locked out entirely ──
DROP POLICY IF EXISTS "Organization members can read claim evidence" ON public.claim_evidence;
CREATE POLICY "Organization members can read claim evidence"
    ON public.claim_evidence FOR SELECT
    USING (public.is_org_member(public.claim_org(claim_id)));

-- ── verification_timeline: previously locked out entirely ──
DROP POLICY IF EXISTS "Organization members can read verification timeline" ON public.verification_timeline;
CREATE POLICY "Organization members can read verification timeline"
    ON public.verification_timeline FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = verification_timeline.document_id AND public.is_org_member(d.organization_id)
    ));

-- ── audit_trails: previously locked out entirely. Read-only to members; append-only by design ──
DROP POLICY IF EXISTS "Organization members can read audit trails" ON public.audit_trails;
CREATE POLICY "Organization members can read audit trails"
    ON public.audit_trails FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = audit_trails.document_id AND public.is_org_member(d.organization_id)
    ));

-- Audit rows must never be edited or removed by a client, only appended by the service role.
DROP POLICY IF EXISTS "Audit trails are immutable" ON public.audit_trails;
REVOKE UPDATE, DELETE ON public.audit_trails FROM anon, authenticated;

-- ── claims: read policy existed, write policies did not ──
DROP POLICY IF EXISTS "Organization members can update claims" ON public.claims;
CREATE POLICY "Organization members can update claims"
    ON public.claims FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = claims.document_id AND public.has_org_role(d.organization_id, ARRAY['admin', 'reviewer'])
    ));

-- ── documents: allow owners/admins to delete for retention and GDPR erasure ──
DROP POLICY IF EXISTS "Organization admins can delete documents" ON public.documents;
CREATE POLICY "Organization admins can delete documents"
    ON public.documents FOR DELETE
    USING (public.has_org_role(organization_id, ARRAY['admin']));

-- ── review_tasks: reviewers could read only tasks already assigned to them, so an unassigned
--    task was invisible to everyone and could never be picked up ──
DROP POLICY IF EXISTS "Reviewers can view review tasks" ON public.review_tasks;
DROP POLICY IF EXISTS "Organization reviewers can view review tasks" ON public.review_tasks;
CREATE POLICY "Organization reviewers can view review tasks"
    ON public.review_tasks FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = review_tasks.document_id AND public.is_org_member(d.organization_id)
    ));

DROP POLICY IF EXISTS "Organization reviewers can claim review tasks" ON public.review_tasks;
CREATE POLICY "Organization reviewers can claim review tasks"
    ON public.review_tasks FOR UPDATE
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = review_tasks.document_id AND public.has_org_role(d.organization_id, ARRAY['admin', 'reviewer'])
    ));

-- ── review_decisions: had an INSERT policy but no SELECT, so a recorded decision was
--    write-only and could never be displayed ──
DROP POLICY IF EXISTS "Organization members can read review decisions" ON public.review_decisions;
CREATE POLICY "Organization members can read review decisions"
    ON public.review_decisions FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = review_decisions.document_id AND public.is_org_member(d.organization_id)
    ));

-- ── verification_jobs: RLS enabled with no policy at all ──
DROP POLICY IF EXISTS "Organization members can read verification jobs" ON public.verification_jobs;
CREATE POLICY "Organization members can read verification jobs"
    ON public.verification_jobs FOR SELECT
    USING (public.is_org_member(organization_id));

-- ── Retention support: find expired documents cheaply ──
CREATE INDEX IF NOT EXISTS documents_retention_until_idx
    ON public.documents(retention_until)
    WHERE retention_until IS NOT NULL;

CREATE INDEX IF NOT EXISTS audit_trails_document_idx ON public.audit_trails(document_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS claim_evidence_claim_idx ON public.claim_evidence(claim_id);
CREATE INDEX IF NOT EXISTS review_decisions_claim_idx ON public.review_decisions(claim_id, created_at DESC);
