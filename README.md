# Research Agent

A LangGraph agent that takes an audience description and research question, collects organic signal across 10 platforms, and returns a structured creative brief for media buyers.

## What it does

Given an audience (e.g. "working class parents in the US") and a research question (e.g. "Are they worried AI will affect their children's job prospects?"), the agent:

1. **Plans** — generates platform-specific search queries tailored to the audience
2. **Collects** — pulls signal from 10 sources in parallel
3. **Gates** — rejects low-volume or off-topic corpora and retries with broader queries
4. **Synthesizes** — produces a structured brief with hooks, angles, verbatim phrases, and media buying signals

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

## Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESEARCH BRIEF
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Audience:  working class parents in the US
Signal:    GREEN — strong, consistent anxiety across Reddit and TikTok
Emotion:   fear / protectiveness
Urgency:   HIGH

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CREATIVE AMMUNITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hook:      "I didn't survive to watch a robot take my kid's shot"
Angle:     Parental sacrifice framing — they worked hard so their kids wouldn't have to
...
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys in .env
```

## Usage

```bash
python3 main.py "audience description" "research question"
```

Example:

```bash
python3 main.py "Muslim immigrants in the US" "Are they anxious about AI taking their jobs?"
```

## Environment variables

```
ANTHROPIC_API_KEY=
SCRAPECREATORS_API_KEY=
XAI_API_KEY=
EXA_API_KEY=
YOUTUBE_API_KEY=
FOREPLAY_API_KEY=
LANGCHAIN_API_KEY=          # optional, for LangSmith tracing
LANGCHAIN_TRACING_V2=true   # optional
LANGCHAIN_PROJECT=research-agent
```

## Architecture

```
plan → [collect_reddit, collect_exa, collect_youtube, collect_foreplay,
         collect_twitter, collect_tiktok, collect_instagram, collect_threads,
         collect_hn, collect_github] → quality_gate → synthesize
                                              ↓ (fail + retry)
                                            plan
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain Anthropic](https://github.com/langchain-ai/langchain).
