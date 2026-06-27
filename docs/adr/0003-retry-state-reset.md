# Retry passes reset collector results — they do not accumulate

When the quality gate fails and the graph loops back to the plan node, `reddit_posts`, `exa_results`, and `youtube_videos` are wiped before the next pass. Only `retry_count` and `previous_query_failure_reason` carry forward. Accumulating results across retries sounds like more signal but produces a corrupted corpus: retry queries are deliberately wider than the original, so mixing results from two incompatible search strategies means the synthesis node is working across content that was never meant to coexist. The `decision_summary` cannot speak coherently to one audience when half the quotes came from a different search angle.

`query_plan` is explicitly NOT reset on retry. The plan node must be able to read its previous `broadening_note` before writing the next one — that field is the plan node's memory of what it already tried.
