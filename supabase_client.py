import os
from datetime import datetime, timezone

from supabase import create_client

_client = None


def _db():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _client


def save_job(job_id: str, audience: str, question: str) -> None:
    _db().table("research_jobs").insert({
        "id": job_id,
        "audience": audience,
        "question": question,
        "status": "queued",
    }).execute()


def update_job(
    job_id: str,
    status: str,
    brief=None,
    eval_scores=None,
    run_cost=None,
    run_time=None,
) -> None:
    data: dict = {"status": status}
    if brief is not None:
        data["brief"] = brief
    if eval_scores is not None:
        data["eval_scores"] = eval_scores
    if run_cost is not None:
        data["run_cost"] = run_cost
    if run_time is not None:
        data["run_time_seconds"] = run_time
    if status in ("complete", "failed"):
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
    _db().table("research_jobs").update(data).eq("id", job_id).execute()


def get_job(job_id: str) -> dict:
    return _db().table("research_jobs").select("*").eq("id", job_id).single().execute().data
