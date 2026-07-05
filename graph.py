import json
import logging
import os
import urllib.parse
import urllib.request

import yt_dlp
from exa_py import Exa
from googleapiclient.discovery import build
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from circuit_breaker import get_breaker
from state import AgentState

_log = logging.getLogger(__name__)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}

_PLAN_TOOL = {
    "name": "query_plan",
    "description": "Generate per-platform search parameters for the research agent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subreddits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subreddit names, no r/ prefix",
            },
            "reddit_search_query": {"type": "string"},
            "exa_query": {"type": "string"},
            "youtube_terms": {"type": "array", "items": {"type": "string"}},
            "foreplay_query": {
                "type": "string",
                "description": "A concrete product or category name Foreplay can match against real running ads — e.g. 'collagen supplement', 'skincare serum', 'invoicing software'. Not a marketing-strategy sentence or audience description — Foreplay searches ad creative content, not commentary about marketing.",
            },
            "twitter_query": {"type": "string"},
            "tiktok_query": {"type": "string"},
            "instagram_query": {"type": "string"},
            "threads_query": {"type": "string"},
            "hn_query": {"type": "string"},
            "github_query": {"type": "string"},
            "broadening_note": {
                "type": ["string", "null"],
                "description": "null on first pass. On retries: one sentence explaining what was broadened and why.",
            },
        },
        "required": [
            "subreddits",
            "reddit_search_query",
            "exa_query",
            "youtube_terms",
            "foreplay_query",
            "twitter_query",
            "tiktok_query",
            "instagram_query",
            "threads_query",
            "hn_query",
            "github_query",
            "broadening_note",
        ],
    },
}

_SYNTHESIS_TOOL = {
    "name": "brief",
    "description": "Synthesize collected community signal into a structured brief for a media buyer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dominant_emotion": {"type": "string"},
            "verbatim_phrases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "phrase": {"type": "string"},
                        "platform": {"type": "string"},
                        "engagement": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["phrase", "platform", "engagement", "url"],
                },
                "minItems": 5,
                "maxItems": 5,
            },
            "x_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "url": {"type": "string"},
                        "likes": {"type": "string"},
                        "author": {"type": "string"},
                    },
                    "required": ["text", "url", "likes", "author"],
                },
                "maxItems": 5,
            },
            "signal": {"type": "string", "enum": ["GREEN", "RED"]},
            "signal_reasoning": {"type": "string"},
            "decision_summary": {"type": "string"},
            "winning_ad_angle": {"type": "string"},
            "cited_sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "platform": {"type": "string"},
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "engagement": {"type": "string"},
                    },
                    "required": ["platform", "url", "title", "engagement"],
                },
            },
            "audience_description": {"type": "string"},
            "recommended_hook": {"type": "string"},
            "recommended_angle": {"type": "string"},
            "best_platform": {"type": "string"},
            "content_gap": {"type": "string"},
            "urgency": {"type": "string"},
            "suggested_followup": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "winning_ads": {
                "type": "array",
                "items": {"type": "object"},
            },
            "paid_competition": {"type": "string"},
            "paid_gap": {"type": "string"},
            "platform_breakdown": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
        "required": [
            "dominant_emotion",
            "verbatim_phrases",
            "x_signals",
            "signal",
            "signal_reasoning",
            "decision_summary",
            "winning_ad_angle",
            "cited_sources",
            "audience_description",
            "recommended_hook",
            "recommended_angle",
            "best_platform",
            "content_gap",
            "urgency",
            "suggested_followup",
            "winning_ads",
            "paid_competition",
            "paid_gap",
            "platform_breakdown",
        ],
    },
}

_plan_llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(
    [_PLAN_TOOL], tool_choice={"type": "tool", "name": "query_plan"}
)

_sonnet = ChatAnthropic(model="claude-sonnet-4-6")
_synthesis_llm = _sonnet.bind_tools(
    [_SYNTHESIS_TOOL], tool_choice={"type": "tool", "name": "brief"}
)

_MAX_RETRIES = 2

_exa = Exa(api_key=os.environ["EXA_API_KEY"])

_sc_headers = {
    "x-api-key": os.environ["SCRAPECREATORS_API_KEY"],
    "Content-Type": "application/json",
}


def _sc_get(url: str, params: dict) -> dict:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full_url, headers=_sc_headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

_youtube = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"])


def plan(state: AgentState) -> dict:
    audience = state["audience_description"]
    question = state["research_question"]
    retry_count = state["retry_count"]
    failure_reason = state.get("previous_query_failure_reason")
    prev_broadening = (state["query_plan"] or {}).get("broadening_note")

    prompt = (
        f"Generate search queries to find organic community signal.\n\n"
        f"Audience: {audience}\n"
        f"Research question: {question}\n\n"
        f"Generate queries for all 10 platforms: 3-5 subreddits (no r/ prefix), "
        f"a Reddit search query, an Exa semantic query, 2-4 YouTube search terms, "
        f"a Twitter/X query, a TikTok query, an Instagram query, a Threads query, "
        f"a Hacker News query, and a GitHub query. "
        f"Set broadening_note to null.\n\n"
        f"For subreddits: always prioritize identity-specific and community-specific "
        f"subreddits over broad topic subreddits.\n\n"
        f"For the Foreplay query: name the actual product or category being advertised "
        f"(e.g. 'collagen supplement', 'invoicing software'), not a sentence about "
        f"marketing strategy or audience framing — Foreplay searches ad creative content, "
        f"not commentary about ads.\n\n"
        f"Rule: if the audience has a specific identity (religion, ethnicity, profession, "
        f"life stage), find the subreddit where that community talks to each other — not "
        f"the subreddit where outsiders talk about the topic.\n\n"
        f"Examples of the principle:\n"
        f"- Muslim audience → r/MuslimLounge not r/religion\n"
        f"- Immigrant audience → r/immigration not r/moving\n"
        f"- Nurse audience → r/nursing not r/healthcare\n"
        f"- Black entrepreneurs → r/Entrepreneur_Resilience not r/entrepreneur\n\n"
        f"Apply this principle to whatever audience you receive. "
        f"Never default to generic topic subreddits when a community-specific one exists."
    )

    if retry_count > 0:
        prompt += (
            f"\n\nThis is retry {retry_count}. Previous queries failed: {failure_reason}."
        )
        if prev_broadening:
            prompt += f"\nPrevious broadening: {prev_broadening}."
        prompt += (
            "\nGenerate meaningfully wider queries — broader subreddits, looser "
            "search terms, wider Foreplay vertical. Set broadening_note to one "
            "sentence explaining what you broadened and why."
        )

    response = _plan_llm.invoke([HumanMessage(content=prompt)])
    return {"query_plan": response.tool_calls[0]["args"]}


_SC_REDDIT = "https://api.scrapecreators.com/v1/reddit"


def collect_reddit(state: AgentState) -> dict:
    def _fetch():
        plan = state["query_plan"]
        query = plan["reddit_search_query"]
        posts = []
        seen_ids: set[str] = set()
        raw = _sc_get(f"{_SC_REDDIT}/search", {"query": query, "sort": "relevance", "timeframe": "month"})
        candidates = raw.get("posts") or raw.get("data") or []
        for sub in plan["subreddits"]:
            if len(candidates) >= 150:
                break
            try:
                r = _sc_get(f"{_SC_REDDIT}/subreddit/search", {"subreddit": sub, "query": query, "sort": "relevance"})
                candidates.extend(r.get("posts") or r.get("data") or [])
            except Exception as exc:
                _log.warning("collect_reddit: subreddit search failed for r/%s: %r", sub, exc)
                continue
        for post in candidates:
            post_id = str(post.get("id", ""))
            if post_id and post_id in seen_ids:
                continue
            if post_id:
                seen_ids.add(post_id)
            permalink = post.get("permalink", "")
            url = f"https://www.reddit.com{permalink}" if permalink else post.get("url", "")
            sub = post.get("subreddit", "")
            if isinstance(sub, dict):
                sub = sub.get("display_name", "")
            posts.append({
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "url": url,
                "score": post.get("score") or post.get("votes") or post.get("ups") or 0,
                "subreddit": sub,
                "comments": [],
            })
        return {"reddit_posts": posts}
    return get_breaker("reddit").call(_fetch) or {"reddit_posts": []}


def collect_exa(state: AgentState) -> dict:
    def _fetch():
        response = _exa.search_and_contents(
            state["query_plan"]["exa_query"],
            highlights=True,
            num_results=10,
        )
        return {"exa_results": [
            {
                "url": r.url,
                "title": r.title,
                "highlights": r.highlights or [],
                "published_date": r.published_date,
            }
            for r in response.results
        ]}
    return get_breaker("exa").call(_fetch) or {"exa_results": []}


def _fetch_transcript(video_id: str) -> str | None:
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        captions = info.get("automatic_captions") or info.get("subtitles") or {}
        en = captions.get("en") or captions.get("en-orig") or []
        sub_url = next((s["url"] for s in en if s.get("ext") == "json3"), None)
        if not sub_url:
            return None
        with urllib.request.urlopen(sub_url, timeout=10) as resp:
            data = json.loads(resp.read())
        parts = [
            seg.get("utf8", "").strip()
            for event in data.get("events", [])
            for seg in event.get("segs", [])
            if seg.get("utf8", "").strip() not in ("", "\n")
        ]
        return " ".join(parts) or None
    except Exception as exc:
        _log.warning("_fetch_transcript failed for video %s: %r", video_id, exc)
        return None


def collect_youtube(state: AgentState) -> dict:
    def _fetch():
        seen_ids: set[str] = set()
        candidates = []
        for term in state["query_plan"]["youtube_terms"]:
            try:
                resp = _youtube.search().list(
                    q=term, part="snippet", type="video", maxResults=10
                ).execute()
                for item in resp.get("items", []):
                    vid_id = item["id"]["videoId"]
                    if vid_id not in seen_ids:
                        seen_ids.add(vid_id)
                        candidates.append({
                            "video_id": vid_id,
                            "title": item["snippet"]["title"],
                            "description": item["snippet"]["description"],
                        })
            except Exception as exc:
                _log.warning("collect_youtube: search failed for term %r: %r", term, exc)
                continue
        if not candidates:
            return {"youtube_videos": []}
        ids_str = ",".join(c["video_id"] for c in candidates)
        stats = _youtube.videos().list(part="statistics", id=ids_str).execute()
        view_counts = {
            item["id"]: int(item["statistics"].get("viewCount", 0))
            for item in stats.get("items", [])
        }
        candidates.sort(key=lambda c: view_counts.get(c["video_id"], 0), reverse=True)
        videos = []
        for i, c in enumerate(candidates[:10]):
            vid_id = c["video_id"]
            videos.append({
                "video_id": vid_id,
                "title": c["title"],
                "url": f"https://www.youtube.com/watch?v={vid_id}",
                "view_count": view_counts.get(vid_id, 0),
                "description": c["description"],
                "transcript": _fetch_transcript(vid_id) if i < 3 else None,
            })
        return {"youtube_videos": videos}
    return get_breaker("youtube").call(_fetch) or {"youtube_videos": []}


def collect_foreplay(state: AgentState) -> dict:
    def _fetch():
        url = "https://public.api.foreplay.co/api/discovery/ads?" + urllib.parse.urlencode({
            "query": state["query_plan"]["foreplay_query"],
            "order": "longest_running",
            "live": "true",
            "limit": 25,
        })
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {os.environ['FOREPLAY_API_KEY']}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        items = data.get("data") or (data if isinstance(data, list) else [])
        return {"foreplay_ads": [
            {
                "ad_id": ad.get("id"),
                "hook_text": ad.get("headline") or ad.get("description"),
                "emotional_driver": max(ad["emotional_drivers"], key=ad["emotional_drivers"].get) if ad.get("emotional_drivers") else None,
                "days_running": (ad.get("running_duration") or {}).get("days"),
                "brand": ad.get("name"),
                "url": ad.get("link_url") or ad.get("foreplay_url"),
            }
            for ad in items
            if ad.get("headline") or ad.get("description")
        ]}
    return get_breaker("foreplay").call(_fetch) or {"foreplay_ads": []}


_XAI_URL = "https://api.x.ai/v1/responses"
_XAI_MODEL = "grok-3"

def collect_twitter(state: AgentState) -> dict:
    def _fetch():
        payload = json.dumps({
            "model": _XAI_MODEL,
            "tools": [{"type": "x_search"}],
            "input": [{
                "role": "user",
                "content": (
                    f"Search X for posts about: {state['query_plan']['twitter_query']}\n"
                    f"Return the exact verbatim text of each post — do not summarize or paraphrase. "
                    f"Return JSON only: {{\"items\": [{{\"text\": \"exact post text verbatim\", \"url\": \"...\", "
                    f"\"like_count\": 0, \"retweet_count\": 0}}]}}"
                ),
            }],
        }).encode()
        req = urllib.request.Request(
            _XAI_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {os.environ['XAI_API_KEY']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        output_text = ""
        for block in data.get("output", []):
            for part in block.get("content", []):
                if part.get("type") == "output_text":
                    output_text += part.get("text", "")
        parsed = _extract_json(output_text)
        if not parsed:
            _log.warning("collect_twitter: could not parse Grok output: %r", output_text[:500])
        return {"twitter_posts": [
            {
                "text": item.get("text", ""),
                "url": item.get("url", ""),
                "like_count": item.get("like_count") or (item.get("engagement") or {}).get("likes") or 0,
                "retweet_count": item.get("retweet_count") or (item.get("engagement") or {}).get("reposts") or 0,
            }
            for item in (parsed.get("items") or [])[:15]
        ]}
    return get_breaker("twitter").call(_fetch) or {"twitter_posts": []}


def collect_tiktok(state: AgentState) -> dict:
    def _fetch():
        data = _sc_get(
            "https://api.scrapecreators.com/v1/tiktok/search/keyword",
            {"query": state["query_plan"]["tiktok_query"], "sort_by": "relevance"},
        )
        videos = []
        for entry in (data.get("search_item_list") or data.get("data") or [])[:15]:
            item = entry.get("aweme_info", entry) if isinstance(entry, dict) else entry
            stats = item.get("statistics") or {}
            video_id = str(item.get("aweme_id", ""))
            share_url = item.get("share_url", "")
            author = item.get("author") or {}
            handle = author.get("unique_id", "") if isinstance(author, dict) else ""
            url = share_url.split("?")[0] if share_url else (
                f"https://www.tiktok.com/@{handle}/video/{video_id}" if handle and video_id else ""
            )
            videos.append({
                "video_id": video_id,
                "description": item.get("desc", ""),
                "play_count": stats.get("play_count", 0),
                "like_count": stats.get("digg_count", 0),
                "url": url,
            })
        return {"tiktok_videos": videos}
    return get_breaker("tiktok").call(_fetch) or {"tiktok_videos": []}


def collect_instagram(state: AgentState) -> dict:
    def _fetch():
        data = _sc_get(
            "https://api.scrapecreators.com/v2/instagram/reels/search",
            {"query": state["query_plan"]["instagram_query"]},
        )
        raw_items = (
            data.get("reels") or data.get("data") or data.get("items")
            or (data if isinstance(data, list) else [])
        )
        posts = []
        for raw in raw_items[:15]:
            caption = raw.get("caption", "")
            if isinstance(caption, dict):
                caption = caption.get("text", "")
            shortcode = raw.get("shortcode") or raw.get("code", "")
            posts.append({
                "post_id": str(raw.get("id") or raw.get("pk", "")),
                "caption": caption,
                "like_count": raw.get("like_count") or 0,
                "view_count": (
                    raw.get("video_play_count") or raw.get("video_view_count")
                    or raw.get("play_count") or 0
                ),
                "url": raw.get("url") or (
                    f"https://www.instagram.com/reel/{shortcode}" if shortcode else ""
                ),
            })
        return {"instagram_posts": posts}
    return get_breaker("instagram").call(_fetch) or {"instagram_posts": []}


def collect_threads(state: AgentState) -> dict:
    def _fetch():
        data = _sc_get(
            "https://api.scrapecreators.com/v1/threads/search",
            {"query": state["query_plan"]["threads_query"]},
        )
        raw_items = (
            data.get("items") or data.get("data") or data.get("threads")
            or data.get("posts") or data.get("search_results") or []
        )
        posts = []
        for raw in raw_items[:15]:
            text = raw.get("text") or raw.get("caption") or raw.get("content") or ""
            if isinstance(text, dict):
                text = text.get("text", "")
            code = raw.get("code") or raw.get("shortcode", "")
            posts.append({
                "post_id": str(raw.get("id") or raw.get("pk") or raw.get("code", "")),
                "text": text,
                "like_count": raw.get("like_count") or raw.get("likes") or 0,
                "url": raw.get("url") or raw.get("share_url") or (
                    f"https://www.threads.net/t/{code}" if code else ""
                ),
            })
        return {"threads_posts": posts}
    return get_breaker("threads").call(_fetch) or {"threads_posts": []}


def collect_hn(state: AgentState) -> dict:
    def _fetch():
        url = "https://hn.algolia.com/api/v1/search?" + urllib.parse.urlencode({
            "query": state["query_plan"]["hn_query"],
            "tags": "story",
            "hitsPerPage": 15,
        })
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        stories = []
        for hit in data.get("hits", []):
            obj_id = hit.get("objectID", "")
            stories.append({
                "story_id": obj_id,
                "title": hit.get("title", ""),
                "text": (hit.get("story_text") or "")[:300],
                "points": hit.get("points") or 0,
                "num_comments": hit.get("num_comments") or 0,
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={obj_id}",
            })
        return {"hn_stories": stories}
    return get_breaker("hn").call(_fetch) or {"hn_stories": []}


def collect_github(state: AgentState) -> dict:
    def _fetch():
        url = "https://api.github.com/search/issues?" + urllib.parse.urlencode({
            "q": state["query_plan"]["github_query"],
            "sort": "reactions",
            "per_page": 15,
        })
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "research-agent"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return {"github_items": [
            {
                "issue_id": issue.get("number"),
                "title": issue.get("title", ""),
                "body": (issue.get("body") or "")[:300],
                "reactions": (issue.get("reactions") or {}).get("total_count", 0),
                "comments": issue.get("comments", 0),
                "url": issue.get("html_url", ""),
                "repo": issue.get("repository_url", "").split("/repos/", 1)[-1],
            }
            for issue in data.get("items", [])
        ]}
    return get_breaker("github").call(_fetch) or {"github_items": []}


def _queries_tried(state: AgentState) -> list[str]:
    plan = state.get("query_plan") or {}
    return [v for v in plan.values() if isinstance(v, str) and v]


def _corpus_sample(state: AgentState) -> str:
    parts = []
    for p in state["reddit_posts"][:5]:
        parts.append(f"[Reddit] {p.get('title','')} — {(p.get('selftext') or '')[:200]}")
    for r in state["exa_results"][:3]:
        parts.append(f"[Web] {r.get('title','')} — {' | '.join((r.get('highlights') or [])[:2])}")
    for v in state["youtube_videos"][:3]:
        parts.append(f"[YouTube] {v.get('title','')} — {(v.get('transcript') or v.get('description') or '')[:200]}")
    for ad in state["foreplay_ads"][:3]:
        parts.append(f"[Ad] {ad.get('hook_text') or ''}")
    for p in state["twitter_posts"][:3]:
        parts.append(f"[Twitter] {(p.get('text') or '')[:200]}")
    for v in state["tiktok_videos"][:3]:
        parts.append(f"[TikTok] {(v.get('description') or '')[:200]}")
    for p in state["instagram_posts"][:3]:
        parts.append(f"[Instagram] {(p.get('caption') or '')[:200]}")
    for p in state["threads_posts"][:3]:
        parts.append(f"[Threads] {(p.get('text') or '')[:200]}")
    for s in state["hn_stories"][:3]:
        parts.append(f"[HN] {s.get('title','')} — {(s.get('text') or '')[:200]}")
    for item in state["github_items"][:3]:
        parts.append(f"[GitHub] {item.get('title','')} — {(item.get('body') or '')[:200]}")
    return "\n".join(parts)


def quality_gate(state: AgentState) -> dict:
    total = sum(len(state[k]) for k in [
        "reddit_posts", "exa_results", "youtube_videos", "foreplay_ads",
        "twitter_posts", "tiktok_videos", "instagram_posts", "threads_posts",
        "hn_stories", "github_items",
    ])
    retry_count = state["retry_count"]

    if total < 10:
        if retry_count >= _MAX_RETRIES:
            return {"error": {
                "reason": f"Volume floor not met after {retry_count + 1} attempts ({total} results).",
                "queries_tried": _queries_tried(state),
                "retry_count": retry_count,
                "suggestion": "Try a broader audience description or more general research question.",
                "failed_at": "volume",
            }, "quality_volume_passed": False, "quality_relevance_passed": False}
        return {
            "quality_volume_passed": False,
            "quality_relevance_passed": False,
            "retry_count": retry_count + 1,
            "previous_query_failure_reason": f"Only {total} results collected (floor is 10).",
        }

    prompt = (
        f"Audience: {state['audience_description']}\n"
        f"Research question: {state['research_question']}\n\n"
        f"Collected content:\n{_corpus_sample(state)}\n\n"
        "Does this content meaningfully address the research question for the stated audience? "
        "Reply PASS or FAIL followed by one sentence explaining why."
    )
    response = _sonnet.invoke([HumanMessage(content=prompt)])
    passed = response.content.strip().upper().startswith("PASS")

    if passed:
        return {"quality_volume_passed": True, "quality_relevance_passed": True}

    reason = response.content.strip()
    if retry_count >= _MAX_RETRIES:
        return {"error": {
            "reason": f"Relevance check failed after {retry_count + 1} attempts: {reason}",
            "queries_tried": _queries_tried(state),
            "retry_count": retry_count,
            "suggestion": "Rephrase the research question to match language the community actually uses.",
            "failed_at": "relevance",
        }, "quality_volume_passed": True, "quality_relevance_passed": False}
    return {
        "quality_volume_passed": True,
        "quality_relevance_passed": False,
        "retry_count": retry_count + 1,
        "previous_query_failure_reason": reason,
    }


def _full_corpus(state: AgentState) -> str:
    parts = []
    for p in state["reddit_posts"]:
        comments = " | ".join((c.get("body") or '')[:100] for c in p.get("comments", []))
        parts.append(f"[Reddit r/{p.get('subreddit','')}] {p.get('title','')} — {(p.get('selftext') or '')[:300]}\nComments: {comments}")
    for r in state["exa_results"]:
        parts.append(f"[Web: {r.get('url','')}] {r.get('title','')} — {' | '.join(r.get('highlights') or [])}")
    for v in state["youtube_videos"]:
        transcript = (v.get("transcript") or v.get("description") or "")[:500]
        parts.append(f"[YouTube: {v.get('url','')}] {v.get('title','')} (views: {v.get('view_count',0)}) — {transcript}")
    for ad in state["foreplay_ads"]:
        parts.append(f"[Ad ({ad.get('brand') or ''}): {ad.get('url') or ''}] hook: {ad.get('hook_text') or ''} | driver: {ad.get('emotional_driver') or ''} | {ad.get('days_running') or ''} days running")
    for p in state["twitter_posts"]:
        parts.append(f"[Twitter: {p.get('url') or ''}] {p.get('text') or ''} (likes: {p.get('like_count',0)})")
    for v in state["tiktok_videos"]:
        parts.append(f"[TikTok: {v.get('url') or ''}] {v.get('description') or ''} (plays: {v.get('play_count',0)})")
    for p in state["instagram_posts"]:
        parts.append(f"[Instagram: {p.get('url') or ''}] {(p.get('caption') or '')[:300]} (views: {p.get('view_count',0)})")
    for p in state["threads_posts"]:
        parts.append(f"[Threads: {p.get('url') or ''}] {(p.get('text') or '')[:300]} (likes: {p.get('like_count',0)})")
    for s in state["hn_stories"]:
        parts.append(f"[HN: {s.get('url') or ''}] {s.get('title') or ''} (points: {s.get('points',0)}) — {(s.get('text') or '')[:300]}")
    for item in state["github_items"]:
        parts.append(f"[GitHub: {item.get('url') or ''}] {item.get('repo') or ''} #{item.get('issue_id','')} {item.get('title') or ''} — {(item.get('body') or '')[:300]}")
    return "\n\n".join(parts)


_SYSTEM_PROMPT = """You are a senior strategist at a world-class performance marketing agency. You turn raw community signal into actionable creative direction that a media buyer can execute today without interpretation.

Your output is read by a CMO in 90 seconds and handed to their team with a clear directive. Every field must be scannable. No paragraphs except where specified.

Field instructions:

dominant_emotion: single word only. Fear. Hope. Anger. Anxiety.

audience_description: who is loud, where, and what demographic signal is visible. Two sentences max.

verbatim_phrases: exactly 5. Must be verbatim — exact words from the corpus, not paraphrased. Reddit, Exa/web, YouTube, TikTok, Instagram, Threads, HN, and GitHub only — never Twitter/X. Include platform, engagement metric, and URL for each. Prioritize highest engagement.

x_signals: up to 5 practitioner signals from Twitter/X only. These are Grok-synthesized — a distilled summary of the post, not a verbatim quote. For each: the insight from the tweet (text), tweet URL, like count (likes), and author handle (author). If no Twitter data was collected, return an empty list.

recommended_hook: write the actual hook. Copy-ready, under 15 words, first person or direct address. Not a description of a hook — the hook itself.

recommended_angle: one sentence. The creative direction. Emotion arc + anchor + CTA direction.

winning_ad_angle: the single strongest angle to test. One sentence.

winning_ads: top 3 from Foreplay data only. If Foreplay returned no results write empty list.

paid_competition: HIGH if 5+ brands running similar angles. MEDIUM if 2-4. LOW if 1. NONE if zero. Be specific.

paid_gap: one sentence. The untapped angle no paid ad is currently running. If no gap exists say so.

best_platform: single platform name. Where signal volume AND engagement are highest combined. Back it with one number.

platform_breakdown: every platform that returned results. Include volume, top content title, engagement number, and signal quality assessment (strong/moderate/thin).

content_gap: the specific question the community is asking loudly that no brand is answering. One sentence, specific, actionable.

urgency: EVERGREEN if pain is structural and ongoing. SPIKE if tied to a news cycle. BUILDING if volume is increasing. One word + one sentence explanation.

suggested_followup: two follow-up research questions that would sharpen this brief. Specific audience + question format.

decision_summary: one paragraph, plain English, written for someone reading on WhatsApp.

signal: GREEN if community is loud, consistent, on-topic for this audience. RED if content exists but doesn't confirm the premise. Be decisive.

signal_reasoning: two sentences max. Why GREEN or RED. Cite a specific data point.

cited_sources: platform, url, title, and one engagement metric per source you quote."""


def _is_twitter_url(url: str) -> bool:
    return "x.com/" in url or "twitter.com/" in url


def synthesize(state: AgentState) -> dict:
    prompt = (
        f"Audience: {state['audience_description']}\n"
        f"Research question: {state['research_question']}\n\n"
        f"Collected signal:\n{_full_corpus(state)}"
    )
    response = _synthesis_llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    brief = response.tool_calls[0]["args"]

    # ponytail: Sonnet ignores platform restriction in verbatim_phrases — enforce in code.
    # Move any Twitter/X entries from verbatim_phrases to x_signals (schema conversion: phrase→text).
    twitter_spill = [p for p in brief.get("verbatim_phrases", []) if _is_twitter_url(p.get("url", ""))]
    if twitter_spill:
        brief["verbatim_phrases"] = [p for p in brief["verbatim_phrases"] if not _is_twitter_url(p.get("url", ""))]
        existing = brief.get("x_signals") or []
        seen_urls = {s.get("url") for s in existing}
        for p in twitter_spill:
            if p.get("url") not in seen_urls:
                existing.append({
                    "text": p.get("phrase", ""),
                    "url": p.get("url", ""),
                    "likes": p.get("engagement", ""),
                    "author": "",
                })
        brief["x_signals"] = existing

    return {"brief": brief}


def _route(state: AgentState) -> str:
    if state.get("error"):
        return "__end__"
    if state["quality_volume_passed"] and state["quality_relevance_passed"]:
        return "synthesize"
    return "plan"


builder = StateGraph(AgentState)

builder.add_node("plan", plan)
builder.add_node("collect_reddit", collect_reddit)
builder.add_node("collect_exa", collect_exa)
builder.add_node("collect_youtube", collect_youtube)
builder.add_node("collect_foreplay", collect_foreplay)
builder.add_node("collect_twitter", collect_twitter)
builder.add_node("collect_tiktok", collect_tiktok)
builder.add_node("collect_instagram", collect_instagram)
builder.add_node("collect_threads", collect_threads)
builder.add_node("collect_hn", collect_hn)
builder.add_node("collect_github", collect_github)
builder.add_node("quality_gate", quality_gate)
builder.add_node("synthesize", synthesize)

builder.set_entry_point("plan")

builder.add_edge("plan", "collect_reddit")
builder.add_edge("plan", "collect_exa")
builder.add_edge("plan", "collect_youtube")
builder.add_edge("plan", "collect_foreplay")
builder.add_edge("plan", "collect_twitter")
builder.add_edge("plan", "collect_tiktok")
builder.add_edge("plan", "collect_instagram")
builder.add_edge("plan", "collect_threads")
builder.add_edge("plan", "collect_hn")
builder.add_edge("plan", "collect_github")

builder.add_edge("collect_reddit", "quality_gate")
builder.add_edge("collect_exa", "quality_gate")
builder.add_edge("collect_youtube", "quality_gate")
builder.add_edge("collect_foreplay", "quality_gate")
builder.add_edge("collect_twitter", "quality_gate")
builder.add_edge("collect_tiktok", "quality_gate")
builder.add_edge("collect_instagram", "quality_gate")
builder.add_edge("collect_threads", "quality_gate")
builder.add_edge("collect_hn", "quality_gate")
builder.add_edge("collect_github", "quality_gate")

builder.add_conditional_edges("quality_gate", _route)

builder.add_edge("synthesize", END)

graph = builder.compile()
