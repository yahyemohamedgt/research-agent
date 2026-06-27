# Reimplement collectors as LangGraph nodes — do not call /last30days

The `/last30days` Claude Code skill is proven for interactive research but is a 1400-line skill contract with a setup wizard, browser cookies, TikTok enrichment pipeline, and badge-formatting logic baked in. Calling it from a LangGraph node via subprocess would couple the product to a Claude Code-specific runtime contract that was never designed to be a library. Instead, the agent reimplements the three signal sources (PRAW, Exa, yt-dlp + YouTube Data API) as clean, typed LangGraph nodes that write structured output into shared state. The `/last30days` skill remains the interactive research tool in Claude Code; the agent is the product layer.

## Considered options

- **Subprocess call to `last30days.py --emit=compact --agent`** — rejected because the skill contract (wizard, cookie extraction, badge output, TikTok enrichment) is not stable as an API surface. Any change to the skill would break the agent silently.
- **Reimplement collectors as LangGraph nodes** — accepted. Same underlying sources, clean interface, no coupling to the skill runtime.
