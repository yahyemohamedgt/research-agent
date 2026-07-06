import logging
import os
from datetime import datetime, timezone

from supabase import create_client

_log = logging.getLogger(__name__)

_client = None


def _db():
    global _client
    if _client is None:
        _client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _client


def save_job(job_id: str, audience: str, question: str, key_suffix: str | None = None) -> None:
    data = {
        "id": job_id,
        "audience": audience,
        "question": question,
        "status": "queued",
    }
    if key_suffix is not None:
        data["key_suffix"] = key_suffix
    try:
        _db().table("research_jobs").insert(data).execute()
    except Exception:
        # ponytail: schema.sql migration for key_suffix may not be applied yet in this Supabase project.
        # Fall back to saving without it rather than failing the whole job. Remove once the migration
        # has been confirmed live everywhere.
        if "key_suffix" in data:
            _log.warning("insert with key_suffix failed for job %s; retrying without it", job_id)
            data.pop("key_suffix")
            _db().table("research_jobs").insert(data).execute()
        else:
            raise


def update_job(
    job_id: str,
    status: str,
    brief=None,
    eval_scores=None,
    query_plan=None,
    run_cost=None,
    run_time=None,
) -> None:
    data: dict = {"status": status}
    if brief is not None:
        data["brief"] = brief
    if eval_scores is not None:
        data["eval_scores"] = eval_scores
    if query_plan is not None:
        data["query_plan"] = query_plan
    if run_cost is not None:
        data["run_cost"] = run_cost
    if run_time is not None:
        data["run_time_seconds"] = run_time
    if status in ("complete", "failed"):
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
    _db().table("research_jobs").update(data).eq("id", job_id).execute()


def get_job(job_id: str) -> dict:
    return _db().table("research_jobs").select("*").eq("id", job_id).single().execute().data


def count_running_jobs() -> int:
    result = _db().table("research_jobs").select("id", count="exact").eq("status", "running").execute()
    return result.count or 0
