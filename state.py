from typing import Literal, TypedDict


class CitedSource(TypedDict):
    platform: Literal[
        "reddit", "exa", "youtube", "foreplay",
        "twitter", "tiktok", "instagram", "threads", "hn", "github",
    ]
    url: str
    title: str
    engagement: str


class WinningAd(TypedDict):
    brand: str
    hook_text: str
    emotional_driver: str
    days_running: int
    url: str


class VerbatimPhrase(TypedDict):
    phrase: str
    platform: str
    engagement: str
    url: str


class PlatformBreakdown(TypedDict):
    platform: str
    volume: str
    top_content: str
    engagement: str
    signal_quality: str


class Brief(TypedDict):
    dominant_emotion: str
    verbatim_phrases: list[VerbatimPhrase]
    signal: Literal["GREEN", "RED"]
    signal_reasoning: str
    decision_summary: str
    winning_ad_angle: str
    cited_sources: list[CitedSource]
    audience_description: str
    recommended_hook: str
    recommended_angle: str
    best_platform: str
    content_gap: str
    urgency: str
    suggested_followup: list[str]
    winning_ads: list[WinningAd]
    paid_competition: Literal["HIGH", "MEDIUM", "LOW", "NONE"]
    paid_gap: str
    platform_breakdown: list[PlatformBreakdown]


class AgentError(TypedDict):
    reason: str
    queries_tried: list[str]
    retry_count: int
    suggestion: str
    failed_at: Literal["volume", "relevance"]


class QueryPlan(TypedDict):
    subreddits: list[str]
    reddit_search_query: str
    exa_query: str
    youtube_terms: list[str]
    foreplay_query: str
    twitter_query: str
    tiktok_query: str
    instagram_query: str
    threads_query: str
    hn_query: str
    github_query: str
    broadening_note: str | None


class AgentState(TypedDict):
    # Inputs
    audience_description: str
    research_question: str

    # Plan output
    query_plan: QueryPlan | None

    # Collector outputs — reset on each retry pass
    reddit_posts: list[dict]
    exa_results: list[dict]
    youtube_videos: list[dict]
    foreplay_ads: list[dict]
    twitter_posts: list[dict]
    tiktok_videos: list[dict]
    instagram_posts: list[dict]
    threads_posts: list[dict]
    hn_stories: list[dict]
    github_items: list[dict]

    # Quality gate
    quality_volume_passed: bool | None
    quality_relevance_passed: bool | None

    # Retry tracking
    retry_count: int
    previous_query_failure_reason: str | None

    # Outputs
    brief: Brief | None
    error: AgentError | None
