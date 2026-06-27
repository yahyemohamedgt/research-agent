import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel

from circuit_breaker import all_statuses
from eval import run_eval
from graph import graph
from supabase_client import get_job, save_job, update_job

app = FastAPI()


class ResearchRequest(BaseModel):
    audience: str
    question: str


# ponytail: mirrors main.py's initial state — keep in sync if AgentState fields change
_BLANK_STATE: dict = {
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
}


async def _run_job(job_id: str, audience: str, question: str) -> None:
    # ponytail: supabase calls are sync — wrap with asyncio.to_thread if concurrency becomes an issue
    start = time.monotonic()
    update_job(job_id, status="running")
    try:
        # Call graph directly (not run_agent) so we keep final_state for eval
        final_state = await graph.ainvoke({
            "audience_description": audience,
            "research_question": question,
            **_BLANK_STATE,
        })
        elapsed = round(time.monotonic() - start, 2)
        result = final_state["error"] if final_state["error"] else final_state["brief"]

        if "failed_at" in result:
            update_job(
                job_id,
                status="failed",
                eval_scores={"failed_at": result["failed_at"], "reason": result.get("reason")},
                run_time=elapsed,
            )
        else:
            eval_scores = None
            try:
                eval_scores = run_eval(result, final_state)
            except Exception:
                pass  # eval failure never blocks brief delivery
            update_job(
                job_id,
                status="complete",
                brief=result,
                eval_scores=eval_scores,
                # ponytail: run_cost is None until graph exposes token usage
                run_time=elapsed,
            )
    except Exception as e:
        elapsed = round(time.monotonic() - start, 2)
        update_job(job_id, status="failed", eval_scores={"error": str(e)}, run_time=elapsed)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0"}


@app.get("/health/collectors")
async def health_collectors():
    return {"collectors": all_statuses()}


@app.post("/research")
async def create_research(body: ResearchRequest):
    job_id = str(uuid4())
    save_job(job_id, body.audience, body.question)
    asyncio.create_task(_run_job(job_id, body.audience, body.question))
    return {"job_id": job_id, "status": "queued"}


@app.get("/research/{job_id}")
async def get_research(job_id: str):
    job = get_job(job_id)
    created = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
    if job.get("completed_at"):
        completed = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
        elapsed = round((completed - created).total_seconds(), 2)
    else:
        elapsed = round((datetime.now(timezone.utc) - created).total_seconds(), 2)
    return {
        "status": job["status"],
        "brief": job.get("brief"),
        "elapsed_seconds": elapsed,
    }
