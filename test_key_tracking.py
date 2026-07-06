from dotenv import load_dotenv

load_dotenv()

from fastapi import HTTPException

import supabase_client
from api import _API_KEYS, _require_api_key

_API_KEYS.clear()
_API_KEYS.update({"validkey1234", "anotherkey5678"})

assert _require_api_key("validkey1234") == "1234"
assert _require_api_key("anotherkey5678") == "5678"

try:
    _require_api_key("wrongkey")
    assert False, "should have raised"
except HTTPException as e:
    assert e.status_code == 401


class _FakeTable:
    def __init__(self, fail_once):
        self.calls = []
        self._fail_once = fail_once

    def insert(self, data):
        self.calls.append(dict(data))
        if self._fail_once and "key_suffix" in data and len(self.calls) == 1:
            raise Exception('column "key_suffix" does not exist')
        return self

    def execute(self):
        return None


class _FakeDB:
    def __init__(self, fail_once):
        self._table = _FakeTable(fail_once)

    def table(self, name):
        return self._table


# column not migrated yet -> falls back to inserting without key_suffix
fake_db = _FakeDB(fail_once=True)
supabase_client._client = fake_db
supabase_client.save_job("job-1", "aud", "q", key_suffix="abcd")
assert len(fake_db._table.calls) == 2
assert fake_db._table.calls[0]["key_suffix"] == "abcd"
assert "key_suffix" not in fake_db._table.calls[1]

# column present -> single insert, key_suffix stored
fake_db = _FakeDB(fail_once=False)
supabase_client._client = fake_db
supabase_client.save_job("job-2", "aud", "q", key_suffix="wxyz")
assert len(fake_db._table.calls) == 1
assert fake_db._table.calls[0]["key_suffix"] == "wxyz"

# no key_suffix passed -> field omitted entirely, no behavior change for old callers
fake_db = _FakeDB(fail_once=False)
supabase_client._client = fake_db
supabase_client.save_job("job-3", "aud", "q")
assert len(fake_db._table.calls) == 1
assert "key_suffix" not in fake_db._table.calls[0]

supabase_client._client = None

print("ok")
