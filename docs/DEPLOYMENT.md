# TruthLens AI — Deployment Guide

## Before you deploy: the quota reality

`gemini-2.5-flash` allows **20 requests/day** on the free tier. Verification makes **two** model
calls, so a free key sustains roughly **10 verifications/day**. Self-check adds a third; a
three-model benchmark costs six; a ten-document batch consumes an entire day.

This was measured against a live key, not read from documentation. **Enable billing before any
demonstration**, or you will hit a 429 mid-presentation. The failure is graceful — a clear error
with the measured stages preserved — but it is still a failure.

`gemini-2.0-*` models report `limit: 0` on the free tier: they are unusable, which is why they are
absent from the `BENCHMARK_MODELS` default.

---

## 1 · Demo-only deployment (fastest path)

Verification works for anyone, immediately. Nothing is stored; the dashboard, review queue and
audit trail explain that they need an account.

```bash
vercel env add GEMINI_API_KEY production      # from https://aistudio.google.com/apikey
vercel --prod
```

Verify: `curl https://<your-app>/api/health` → `"status": "healthy"`, persistence mode `demo-only`.

---

## 2 · Authentication (Google sign-in)

Guests get demo mode with no setup. Storage, dashboard, review queue and audit need an account.

1. In your Supabase project: **Authentication → Providers → Google**, enable it, and paste a
   Google OAuth client id and secret from the Google Cloud console.
2. Add your app origin to **Authentication → URL Configuration → Redirect URLs**
   (`http://localhost:3000` for development, plus your production origin).
3. Set the two public client variables:

```bash
vercel env add VITE_SUPABASE_URL       production
vercel env add VITE_SUPABASE_ANON_KEY  production
```

The anon key is publishable and belongs in the browser bundle. The **service-role key must never
be `VITE_`-prefixed** — that would ship a key that bypasses row-level security to every visitor.

Without these two the app still runs; everyone simply stays in demo mode, and the UI says so.

---

## 3 · Deterministic OCR (PaddleOCR)

PaddleOCR is a Python library with native dependencies. It **cannot run inside Vercel's Node
functions**, so it is a separate service. Without it the pipeline falls back to model transcription
and labels the run `model OCR` — correct, but slower, non-reproducible, and it consumes model quota.

```bash
cd python
pip install -r requirements.txt
uvicorn ocr_service:app --host 0.0.0.0 --port 8100
```

Deploy it anywhere that runs a long-lived Python process — Fly.io, Railway, Render, Cloud Run, or
a container on your own infrastructure. The first request loads the model (several seconds, a few
hundred MB); afterwards it is fast. Then point the API at it:

```bash
vercel env add OCR_SERVICE_URL production   # https://your-ocr-service.example.com
```

Verify with `curl https://your-ocr-service/health` → `{"status":"ok"}`, and confirm a verification
result shows the **PaddleOCR** badge rather than **model OCR**.

---

## 4 · Workspace deployment (full platform)

### 4.1 Provision the database

Any PostgreSQL with PostgREST in front works; Supabase is the tested path.

```bash
# Apply ALL FIVE migrations, in filename order. The second is not optional:
# the first enables RLS on three tables while dropping their policies, which
# locks them out entirely until the second restores access.
supabase db push
```

Or manually:

```bash
for f in supabase/migrations/*.sql; do psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$f"; done
```

All five are idempotent — verified by applying them twice against PostgreSQL 16.

### 4.2 Configure the API

```bash
vercel env add GEMINI_API_KEY              production
vercel env add SUPABASE_URL                production
vercel env add SUPABASE_SERVICE_ROLE_KEY   production
```

The service-role key must **never** be exposed to the browser. It is not prefixed `VITE_`, so Vite
will not bundle it — do not rename it.

### 4.3 Deploy and verify

```bash
vercel --prod
curl https://<your-app>/api/health
```

Expect `"status": "healthy"` and `persistence.mode: "workspace"`. If you get `degraded`, the
database is configured but unreachable — verification still works, storage does not.

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Vision provider credential. Never prefixed `VITE_` |
| `GEMINI_MODEL` | No | Default `gemini-2.5-flash`. Reported to the UI as `modelUsed` |
| `BENCHMARK_MODELS` | No | Comma-separated. Do **not** include `gemini-2.0-*` on free tier |
| `SUPABASE_URL` | No | Presence enables storage for signed-in users |
| `VITE_SUPABASE_URL` | No | Public — enables Google sign-in in the browser |
| `VITE_SUPABASE_ANON_KEY` | No | Public, publishable. Never the service-role key |
| `OCR_SERVICE_URL` | No | PaddleOCR service. Absent → model-transcription fallback |
| `SUPABASE_SERVICE_ROLE_KEY` | No | Required alongside `SUPABASE_URL` |
| `DEFAULT_RETENTION_DAYS` | No | Default 30. Per-account override in Admin → Account |
| `VERIFY_RATE_LIMIT` | No | Default 10 per window |
| `VERIFY_RATE_WINDOW_MS` | No | Default 60000 |
| `VITE_BACKEND` | No | `api` (default) or `python` |

---

## Health checks and monitoring

Point your monitor at `GET /api/health`.

| `status` | HTTP | Action |
|---|---|---|
| `healthy` | 200 | — |
| `degraded` | 200 | Warn. Verification works; storage/review/analytics do not |
| `unhealthy` | 503 | Page. No provider configured — verification impossible |

The response also carries `checks.persistence.latencyMs`, useful as a database latency signal.

**Logs.** Serverless logs go to `vercel logs --prod`. Sanitised errors returned to users carry a
six-character reference that appears in the server log line, so a user-reported failure can be
found without asking them to reproduce it.

**In-app audit.** Admin → Activity shows the API activity log and every human review decision,
scoped to the signed-in account.

---

## Rollback

Vercel keeps every deployment immutable:

```bash
vercel ls                          # find the last good deployment
vercel promote <deployment-url>    # instant, no rebuild
```

**Database rollback is not automatic.** The migrations are additive — new tables and nullable
columns — so an older application build runs against a newer schema without error. That makes
application rollback safe on its own. Reverting a migration requires a hand-written down script;
take a snapshot before applying migrations in production.

---

## Security checklist before going public

- [ ] Billing enabled on the provider key (otherwise the app is trivially DoS-ed by quota)
- [ ] Durable rate limiting in front (Upstash, or a WAF). **The built-in limiter is in-memory and
      per function instance** — a guardrail against one caller hammering a warm instance, not a
      billing control
- [ ] Service-role key set as a Vercel *secret*, not in `vercel.json` or any committed file
- [ ] Security headers verified: `curl -I https://<your-app>` should show CSP, HSTS,
      `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`
- [ ] Retention policy set to your requirement (Admin → Account)
- [ ] Google sign-in configured, and the Supabase redirect URLs include your production origin
- [ ] Understood: there are **no organisation roles**. Each account sees only its own data; there
      is no team sharing or per-role permission model yet

---

## Local development

```bash
npm install
cp .env.example .env        # add GEMINI_API_KEY
npm run dev                 # http://localhost:8080 (frontend only)
npx vercel dev              # frontend + /api functions
npm run verify              # typecheck + lint + test + build
```

For a local database:

```bash
docker run -d --name tl-pg -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=truthlens \
  -p 55432:5432 postgres:16-alpine
for f in supabase/migrations/*.sql; do
  docker exec -i tl-pg psql -U postgres -d truthlens -v ON_ERROR_STOP=1 < "$f"
done
```

## Python backend (optional)

The `python/` FastAPI server serves the legacy RLVO research demos, not the TruthLens pipeline.
It is not required for any TruthLens feature.

```bash
cd python && pip install -r requirements.txt
uvicorn server:app --port 8000
# then set VITE_BACKEND=python
```
