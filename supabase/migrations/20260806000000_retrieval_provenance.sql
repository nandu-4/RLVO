-- ── Retrieval provenance and pre-verification risk ──
-- Migration: 20260806000000_retrieval_provenance.sql
--
-- The verification pipeline now retrieves evidence with an independent search engine before the
-- verifying model sees a claim, and predicts hallucination risk before verification runs. None
-- of that was representable in the schema, so the audit record could not answer two questions an
-- enterprise reviewer will ask: what did the search actually look at, and did the verifier use
-- what it found?

ALTER TABLE public.claim_evidence
    -- Which retrieval strategies surfaced this block (value-match, lexical, numeric, ...).
    ADD COLUMN IF NOT EXISTS retrieved_by TEXT[] NOT NULL DEFAULT '{}',
    -- False when retrieval returned the block but the verifier did not rely on it. Keeping
    -- uncited candidates is deliberate: "what the verifier ignored" is audit-relevant.
    ADD COLUMN IF NOT EXISTS cited BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE public.claims
    ADD COLUMN IF NOT EXISTS evidence_strength INT DEFAULT 0,
    -- How many of the five trust signals were independently measured (0-5). A claim scored on
    -- two signals is not comparable to one scored on five, and the score alone hides that.
    ADD COLUMN IF NOT EXISTS signals_measured INT NOT NULL DEFAULT 0 CHECK (signals_measured BETWEEN 0 AND 5),
    ADD COLUMN IF NOT EXISTS hallucination_risk_level TEXT CHECK (hallucination_risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    ADD COLUMN IF NOT EXISTS hallucination_risk_score INT CHECK (hallucination_risk_score BETWEEN 0 AND 100),
    -- The "why this score exists" narrative, stored so a report can be regenerated identically.
    ADD COLUMN IF NOT EXISTS score_rationale JSONB,
    ADD COLUMN IF NOT EXISTS retrieval_candidates INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS retrieval_cited INT NOT NULL DEFAULT 0;

ALTER TABLE public.documents
    -- Document-level legibility, so a low trust score can be attributed to a poor scan rather
    -- than to the model.
    ADD COLUMN IF NOT EXISTS mean_legibility INT CHECK (mean_legibility BETWEEN 0 AND 100),
    ADD COLUMN IF NOT EXISTS indexed_blocks INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS page_count INT NOT NULL DEFAULT 1;

-- Relationships between claims, derived from where their evidence physically sits.
CREATE TABLE IF NOT EXISTS public.claim_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    from_claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
    to_claim_id UUID NOT NULL REFERENCES public.claims(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('shared-evidence', 'same-region', 'same-page', 'lexical')),
    strength NUMERIC(3,2) NOT NULL DEFAULT 0 CHECK (strength BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT claim_relations_distinct CHECK (from_claim_id <> to_claim_id)
);

ALTER TABLE public.claim_relations ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Organization members can read claim relations" ON public.claim_relations;
CREATE POLICY "Organization members can read claim relations"
    ON public.claim_relations FOR SELECT
    USING (EXISTS (
        SELECT 1 FROM public.documents d
        WHERE d.id = claim_relations.document_id AND public.is_org_member(d.organization_id)
    ));

CREATE INDEX IF NOT EXISTS claim_relations_document_idx ON public.claim_relations(document_id);
CREATE INDEX IF NOT EXISTS claims_risk_idx ON public.claims(hallucination_risk_level) WHERE hallucination_risk_level IS NOT NULL;
