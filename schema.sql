CREATE TABLE research_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  audience TEXT NOT NULL,
  question TEXT NOT NULL,
  status TEXT DEFAULT 'queued',
  brief JSONB,
  eval_scores JSONB,
  run_cost DECIMAL,
  run_time_seconds DECIMAL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);
