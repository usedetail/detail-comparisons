# Review-bot comparison

## Re-run the experiment

```bash
# 1. Fetch bot review comments for the configured EVAL_BOTS / EVAL_REPOS / EVAL_WEEK
#    (queries the ClickHouse playground, writes data/wild_reviews.json)
uv run collect-wild-reviews

# 2. Run the tournament: filter → summarize → judge → BT rank.
#    Cached LLM calls live in data/{filters,summaries,judgments}.json,
#    so re-runs hit cache and are free.
export ANTHROPIC_API_KEY=sk-ant-...
uv run evaluate
```

Configuration: [`src/review_bot_comparison/config.py`](src/review_bot_comparison/config.py).
