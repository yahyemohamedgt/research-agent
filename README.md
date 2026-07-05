# Research Agent

Tells you whether a specific group of people actually feels a certain way about something — in their own words, backed by real sources — and what the paid-ad landscape around it looks like.

## Why this one

Right now, "does this audience actually feel this way" gets answered one of three ways: scroll Reddit and TikTok by hand for a few hours and hope you caught the real conversation, guess off gut feel and ship the ad anyway, or pay for a research deck that lands two weeks after the trend already peaked. All three cost you the thing that decides whether a campaign wins — being first with the right hook, in the audience's own words.

This agent gives you a decisive answer — GREEN or RED, not a paragraph you have to interpret — in about a minute, pulled from ten platforms at once instead of the two or three a person can realistically check by hand. And it doesn't stop at the community: Foreplay's ad data means the same brief also tells you who's already running paid on that angle and where the gap still is. You're not just informed, you're positioned to move before the angle is crowded.

The one thing every AI research tool risks is confidently making things up. This one doesn't get the benefit of the doubt: every verbatim quote it hands you is checked against the raw data it actually collected, and a second, independent model grades the brief before it ever reaches you. If it can't back a claim with a real source, it doesn't get to keep it.

## Who else this is for

This isn't a media-buying tool. It's a lie detector for whether anyone actually wants what you're about to build, market, or sell — and "market" here can mean an ad, a feature, a pitch, or a hire.

**Founders, before they write a line of code.** Three months of build time versus ninety seconds of the truth: is this a real pain, in real words, from real people — or a problem you invented in a doc?

**Sales teams, before the call.** Walk in knowing what the buyer actually said, unprompted, instead of what the deck assumes they care about.

**Anyone sizing up a market.** See who's already there, what they're running, and where they're not — before you guess your way into a category someone else already owns.

Same engine, pointed inward instead of outward — at your own community, your own funnel, your own product roadmap — works exactly the same way.

## What it does

Give it an audience (e.g. "new parents doing sleep training") and a research question (e.g. "are they anxious about doing it wrong?"), and it:

1. **Plans** — generates platform-specific search queries tailored to that exact audience, preferring identity-specific communities over broad topic ones (r/MuslimLounge, not r/religion)
2. **Collects** — pulls signal from 10 sources in parallel (Reddit, Exa, YouTube, Foreplay, Twitter/X, TikTok, Instagram, Threads, HN, GitHub), each isolated behind its own circuit breaker
3. **Gates** — rejects runs with too little data or off-topic data, and retries with meaningfully broader queries (up to 2 retries) before giving up
4. **Synthesizes** — produces a structured brief: dominant emotion, a GREEN/RED signal verdict with reasoning, verbatim community phrases, a copy-ready hook, paid competition and paid gap, platform breakdown, content gap, and cited sources
5. **Evaluates** — checks every verbatim phrase against the raw corpus for hallucinations, and has a second model (GPT-4o-mini) independently score the brief on specificity, evidence, actionability, and hook quality
6. **Persists** — stores the brief, eval scores, query plan, and run time in Supabase, queryable by job ID

The output is meant to be read once, in ninety seconds, and acted on — not interpreted like a dashboard.

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

A single-page frontend is served at `/` (`static/index.html`) — enter your API key (saved to browser `localStorage`, never shipped in the page itself), an audience, and a research question, and it polls the job until it completes or fails. Inline hints and three clickable examples show what a well-formed audience/question pair looks like.

## API

All endpoints except `/health` require an `X-API-Key` header matching one of the comma-separated keys in the `API_KEY` env var. `POST /research` is rate-limited to 5 requests/minute per IP and capped at 5 concurrent running jobs.

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

## CLI (v1)

```bash
python3 main.py "new parents doing sleep training" "are they anxious about doing it wrong?"
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

Configured for Railway — no Dockerfile needed, Nixpacks auto-detects the Python app. `Procfile` sets the start command (`uvicorn api:app --host 0.0.0.0 --port $PORT`). Set all env vars below in the Railway service, then deploy from this repo (GitHub-connected, auto-deploys on push to `main`).

## Environment variables

See `.env.example` for the full list and comments on what each key is for. `API_KEY` accepts a comma-separated list — one or more keys are all valid.

## Stack

LangGraph · LangChain Anthropic · FastAPI · uvicorn · Supabase · OpenAI API

## Architecture

```
plan → [collect_reddit, collect_exa, collect_youtube, collect_foreplay,
         collect_twitter, collect_tiktok, collect_instagram, collect_threads,
         collect_hn, collect_github] → quality_gate → synthesize
                                              ↓ (fail + retry)
                                            plan
```

Each collector is wrapped in a `CircuitBreaker` — on 3 consecutive failures the collector goes OPEN and returns `[]` immediately, allowing the rest of the run to continue with graceful degradation.

The LangGraph graph itself ends at `synthesize`. Eval (hallucination check + judge scoring) and Supabase persistence happen one layer up, in `api.py`'s `_run_job`, which wraps `graph.ainvoke()` — they aren't graph nodes, so don't go looking for them in `graph.py`.

## What's next

- **Corpus persistence.** Right now the raw collector output (all 10 platforms, pre-synthesis) isn't saved anywhere — auditing a hallucination flag weeks later means re-running the query. Next step is a `corpus` JSONB column on `research_jobs`.
- **Real user management.** Multi-key auth (comma-separated `API_KEY`) works for sharing with a handful of people; it doesn't scale past that. A proper keys table (issue/revoke per person, usage tracking) is the natural next step if this grows past friends-and-family.
- **A delivery path beyond the web UI.** `run_agent` was deliberately designed as a pure `Brief | AgentError` function so any caller — CLI, API, a WhatsApp bot — handles it identically ([ADR 0007](docs/adr/0007-agent-error-as-structured-return.md)). The web UI is the first caller; a chat-delivered brief (so it lands where a media buyer actually reads messages) is the logical next one.
- **Atomic concurrency control.** The 5-concurrent-job cap currently reads a count from Supabase, which races under real concurrent load — fine at today's traffic, a known ceiling if usage grows (marked `ponytail:` in `api.py`).
- **Reddit comments are dead code.** `collect_reddit` hardcodes `"comments": []` — it's never populated, yet `_full_corpus` references `p.get("comments", [])` as if it were real data. Reddit comment-level context (arguably the richest part of Reddit) is entirely absent today.
- **TikTok hashtags and engagement breakdown.** `text_extra` (hashtags) and `statistics.comment_count`/`share_count` are available in the raw ScrapeCreators response and currently dropped — only `digg_count`/`play_count` are captured.
- **YouTube Data API daily quota.** The free tier's 10,000 units/day (~100 units per search call) exhausts fast under repeated testing — when it does, `collect_youtube` degrades gracefully (logged, breaker stays closed, YouTube just drops out of the brief) rather than failing the run, but it's worth knowing if YouTube signal looks thin.

---

**Status: v2 — production API with persistence, eval, and multi-key auth.**
