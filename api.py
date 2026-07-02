import asyncio
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from circuit_breaker import all_statuses
from eval import run_eval
from graph import graph
from supabase_client import get_job, save_job, update_job

_API_KEY = os.environ.get("API_KEY", "")
_MAX_CONCURRENT_JOBS = 5
_MAX_INPUT_LENGTH = 500

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _require_api_key(x_api_key: str = Header(...)):
    if not _API_KEY or x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ResearchRequest(BaseModel):
    audience: str
    question: str

    @field_validator("audience", "question")
    @classmethod
    def truncate(cls, v: str) -> str:
        return v[:_MAX_INPUT_LENGTH]


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
    start = time.monotonic()
    update_job(job_id, status="running")
    try:
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
                pass
            update_job(
                job_id,
                status="complete",
                brief=result,
                eval_scores=eval_scores,
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
@limiter.limit("5/minute")
async def create_research(
    request: Request,
    body: ResearchRequest,
    x_api_key: str = Header(...),
):
    _require_api_key(x_api_key)

    # ponytail: Supabase count query — swap for Redis atomic counter if throughput demands it
    from supabase_client import count_running_jobs
    if count_running_jobs() >= _MAX_CONCURRENT_JOBS:
        raise HTTPException(status_code=429, detail="Too many concurrent jobs. Try again shortly.")

    job_id = str(uuid4())
    save_job(job_id, body.audience, body.question)
    asyncio.create_task(_run_job(job_id, body.audience, body.question))
    return {"job_id": job_id, "status": "queued"}


@app.get("/research/{job_id}")
async def get_research(
    job_id: str,
    x_api_key: str = Header(...),
):
    _require_api_key(x_api_key)

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
