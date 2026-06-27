import json
import os

from openai import OpenAI

from state import AgentState

_judge = None


def _get_judge() -> OpenAI:
    global _judge
    if _judge is None:
        _judge = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _judge


def _corpus(state: dict) -> str:
    parts = []
    for p in state.get("reddit_posts", []):
        parts += [p.get("title") or "", p.get("selftext") or ""]
        for c in p.get("comments", []):
            parts.append(c.get("body") or "")
    for r in state.get("exa_results", []):
        for h in r.get("highlights", []):
            parts.append(h or "")
    for v in state.get("youtube_videos", []):
        parts += [v.get("transcript") or "", v.get("description") or ""]
    for t in state.get("tiktok_videos", []):
        parts.append(t.get("description") or "")
    for i in state.get("instagram_posts", []):
        parts.append(i.get("caption") or "")
    for th in state.get("threads_posts", []):
        parts.append(th.get("text") or "")
    for tw in state.get("twitter_posts", []):
        parts.append(tw.get("text") or "")
    for h in state.get("hn_stories", []):
        parts += [h.get("title") or "", h.get("text") or ""]
    for g in state.get("github_items", []):
        parts += [g.get("title") or "", g.get("body") or ""]
    for fa in state.get("foreplay_ads", []):
        parts.append(fa.get("hook_text") or "")
    return " ".join(parts).lower()


def check_hallucinations(brief: dict, state: AgentState) -> list[dict]:
    corpus = _corpus(state)
    results = []
    for phrase in brief.get("verbatim_phrases", []):
        if isinstance(phrase, dict):
            quote = phrase.get("quote") or phrase.get("phrase") or ""
            platform = phrase.get("platform")
        else:
            quote = phrase
            platform = None
        verdict = "VERIFIED" if quote.lower() in corpus else "NOT_FOUND"
        results.append({"quote": quote, "platform": platform, "verdict": verdict})
    return results


def judge_brief(brief: dict) -> dict:
    response = _get_judge().chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, objective evaluator of market research briefs for performance marketing teams. "
                    "Score strictly — a brief that could apply to any audience scores low on specificity. "
                    "Penalize vague claims, generic hooks, and unsupported conclusions."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Score this brief 1-10 on each dimension and return JSON with these exact keys:\n"
                    "- specificity: insights specific to THIS audience or generic observations anyone could make?\n"
                    "- evidence: every claim traceable to real cited sources?\n"
                    "- actionability: media buyer can act on this today without asking follow-up questions?\n"
                    "- hook_quality: recommended hook uses real verbatim community language, not AI-generated copy?\n"
                    "- overall: weighted average of the above\n"
                    "- flags: array of specific issues found\n\n"
                    f"Brief:\n{json.dumps(brief, indent=2)}"
                ),
            },
        ],
    )
    return json.loads(response.choices[0].message.content)


def run_eval(brief: dict, state: AgentState) -> dict:
    hallucination_check = check_hallucinations(brief, state)
    hallucinations_found = sum(1 for h in hallucination_check if h["verdict"] == "NOT_FOUND")
    judge_scores = judge_brief(brief)
    return {
        "hallucination_check": hallucination_check,
        "hallucinations_found": hallucinations_found,
        "judge_scores": judge_scores,
        "eval_passed": hallucinations_found == 0 and judge_scores.get("overall", 0) >= 6,
    }
