import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from graph import graph
from state import AgentError, Brief


async def run_agent(audience: str, question: str) -> Brief | AgentError:
    state = await graph.ainvoke({
        "audience_description": audience,
        "research_question": question,
        "query_plan": None,
        "reddit_posts": [],
        "exa_results": [],
        "youtube_videos": [],
        "foreplay_ads": [],
        "twitter_posts": [],
        "tiktok_videos": [],
        "instagram_posts": [],
        "threads_posts": [],
        "hn_stories": [],
        "github_items": [],
        "quality_volume_passed": None,
        "quality_relevance_passed": None,
        "retry_count": 0,
        "previous_query_failure_reason": None,
        "brief": None,
        "error": None,
    })
    return state["error"] if state["error"] else state["brief"]


def print_brief(brief: dict) -> None:
    divider = "━" * 50

    print(f"\n{divider}")
    print("RESEARCH BRIEF")
    print(divider)
    print(f"Audience:  {brief.get('audience_description', '')}")
    print(f"Signal:    {brief.get('signal')} — {brief.get('signal_reasoning')}")
    print(f"Emotion:   {brief.get('dominant_emotion')}")
    print(f"Urgency:   {brief.get('urgency')}")

    print(f"\n{divider}")
    print("CREATIVE AMMUNITION")
    print(divider)
    print(f"Hook:      {brief.get('recommended_hook')}")
    print(f"Angle:     {brief.get('recommended_angle')}")

    print("\nVerbatim phrases:")
    for i, p in enumerate(brief.get('verbatim_phrases') or [], 1):
        if isinstance(p, dict):
            print(f"  {i}. \"{p.get('phrase')}\"")
            print(f"     {p.get('platform')} · {p.get('engagement')} · {p.get('url')}")
        else:
            print(f"  {i}. \"{p}\"")

    print(f"\n{divider}")
    print("WHERE TO SPEND")
    print(divider)
    print(f"Best platform:    {brief.get('best_platform')}")
    print(f"Paid competition: {brief.get('paid_competition')}")
    print(f"Paid gap:         {brief.get('paid_gap')}")

    winning_ads = brief.get('winning_ads') or []
    if winning_ads:
        print("\nWinning ads (Foreplay):")
        for ad in winning_ads:
            print(f"  · {ad.get('brand')} — {ad.get('days_running')} days")
            print(f"    Hook: {ad.get('hook_text')}")
            print(f"    Emotion: {ad.get('emotional_driver')}")
    else:
        print("\nWinning ads: None found — first mover opportunity")

    print(f"\n{divider}")
    print("PLATFORM BREAKDOWN")
    print(divider)
    for p in brief.get('platform_breakdown') or []:
        print(f"  {p.get('platform'):<12} {p.get('signal_quality'):<10} "
              f"{p.get('volume')} · {p.get('top_content', '')[:50]}")

    print(f"\n{divider}")
    print("CONTENT GAP")
    print(divider)
    print(f"  {brief.get('content_gap')}")

    print(f"\n{divider}")
    print("FOLLOW-UP RESEARCH")
    print(divider)
    for q in brief.get('suggested_followup') or []:
        print(f"  → {q}")

    print(f"\n{divider}")
    print("SOURCES")
    print(divider)
    for s in brief.get('cited_sources') or []:
        print(f"  [{s.get('platform')}] {s.get('title', '')[:60]}")
        print(f"  {s.get('url')} · {s.get('engagement')}")

    print(f"\n{divider}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audience")
    parser.add_argument("question")
    args = parser.parse_args()

    result = asyncio.run(run_agent(args.audience, args.question))

    if "failed_at" in result:
        print(json.dumps(result), file=sys.stderr)
        sys.exit(1)

    print_brief(result)
