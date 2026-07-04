# Research Agent

A LangGraph agent that takes an audience description and research question, collects organic signal across 10 platforms, and returns a structured creative brief for media buyers.

**Status: v2 complete — production API with persistence and eval**

## What it does

Given an audience (e.g. "working class parents in the US") and a research question (e.g. "Are they worried AI will affect their children's job prospects?"), the agent:

1. **Plans** — generates platform-specific search queries tailored to the audience
2. **Collects** — pulls signal from 10 sources in parallel, each protected by a circuit breaker
3. **Gates** — rejects low-volume or off-topic corpora and retries with broader queries
4. **Synthesizes** — produces a structured brief with hooks, angles, verbatim phrases, and media buying signals
5. **Evaluates** — runs a hallucination check against the raw corpus and a GPT-4o-mini quality judge
6. **Persists** — stores every brief, eval scores, and run metadata in Supabase

## Platforms

| Platform | Source |
|---|---|
| Reddit | ScrapeCreators API |
| Twitter/X | xAI Responses API (Grok-3 + x_search) |
| TikTok | ScrapeCreators API |
| Instagram | ScrapeCreators API |
| Threads | ScrapeCreators API |
| YouTube | YouTube Data API v3 |
| Foreplay | Foreplay Public API (winning ads) |
| Exa | Exa semantic search |
| Hacker News | HN Algolia public API |
| GitHub | GitHub issues search API |

## Web UI

A single-page frontend is served at `/` (`static/index.html`) — enter your API key (saved to browser `localStorage`, never shipped in the page itself), an audience, and a research question, and it polls the job until it completes or fails.

## API

All endpoints except `/health` require an `X-API-Key` header matching the `API_KEY` env var. `POST /research` is rate-limited to 5 requests/minute per IP and capped at 5 concurrent running jobs.

```bash
# Start the server
uvicorn api:app --reload

# Submit a research run
POST /research
X-API-Key: <your key>
{"audience": "...", "question": "..."}
→ {"job_id": "uuid", "status": "queued"}

# Poll for results
GET /research/{job_id}
X-API-Key: <your key>
→ {"status": "complete", "brief": {...}, "eval_scores": {...}, "elapsed_seconds": N}
→ {"status": "failed", "brief": null, "eval_scores": {"failed_at": "volume"|"relevance", "reason": "...", "suggestion": "..."}, "elapsed_seconds": N}

# Health (no auth)
GET /health
→ {"status": "ok", "version": "2.0"}

# Collector circuit breaker status (no auth)
GET /health/collectors
→ {"collectors": [{"name": "reddit", "state": "CLOSED", "failures": 0}, ...]}
```

## New in v2

- **FastAPI async API** — `POST /research`, `GET /research/:id`, fire-and-forget background jobs
- **Supabase persistence** — every brief, eval scores, query plan, cost, and run time stored per job
- **Eval layer** — pure-Python hallucination check against the raw corpus + GPT-4o-mini judge scoring specificity, evidence, actionability, and hook quality
- **Circuit breakers** — all 10 collectors protected; OPEN after 3 failures, auto-recovery after 10 min
- **GET /health/collectors** — real-time circuit breaker state for all collectors
- **OpenAI API** — GPT-4o-mini for cross-model eval (Sonnet synthesizes, GPT-4o-mini judges)
- **API key auth + rate limiting** — `X-API-Key` header required, 5 req/min per IP, 5 concurrent job cap
- **Web UI** — static single-page frontend at `/`, no build step

## CLI (v1)

```bash
python3 main.py "audience description" "research question"
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

Run `schema.sql` in your Supabase SQL editor to create the `research_jobs` table.

## Deploy

Configured for Railway — no Dockerfile needed, Nixpacks auto-detects the Python app. `Procfile` sets the start command (`uvicorn api:app --host 0.0.0.0 --port $PORT`). Set all env vars above in the Railway service, then deploy from this repo.

## Environment variables

```
ANTHROPIC_API_KEY=
SCRAPECREATORS_API_KEY=
XAI_API_KEY=
EXA_API_KEY=
YOUTUBE_API_KEY=
FOREPLAY_API_KEY=
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
OPENAI_API_KEY=
API_KEY=                    # required — protects /research endpoints
LANGCHAIN_API_KEY=          # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=true   # optional
LANGCHAIN_PROJECT=research-agent
```

## Stack

LangGraph · LangChain Anthropic · FastAPI · uvicorn · Supabase · OpenAI API

## Architecture

```
plan → [collect_reddit, collect_exa, collect_youtube, collect_foreplay,
         collect_twitter, collect_tiktok, collect_instagram, collect_threads,
         collect_hn, collect_github] → quality_gate → synthesize → eval
                                              ↓ (fail + retry)                    ↓
                                            plan                            supabase_client
```

Each collector is wrapped in a `CircuitBreaker` — on 3 consecutive failures the collector goes OPEN and returns `[]` immediately, allowing the rest of the run to continue with graceful degradation.
