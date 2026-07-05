import logging
import time

_log = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Three states:
    CLOSED   → healthy, calls pass through
    OPEN     → broken, calls return [] immediately
    HALF_OPEN → recovery attempt, one call allowed through
    """

    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: int = 600):
        self.name = name
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: float | None = None
        self.state = "CLOSED"

    def call(self, fn, *args, **kwargs):
        if self.state == "OPEN":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                return []  # Skip silently

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            _log.warning("circuit breaker %r call failed: %r", self.name, exc)
            self._on_failure()
            return []

    def _on_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "OPEN"

    def status(self) -> dict:
        return {"name": self.name, "state": self.state, "failures": self.failures}


def with_backoff(fn, *args, max_retries: int = 3, **kwargs):
    """On failure: wait 1s, 2s, 4s before giving up. Returns [] on final failure — never raises."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return []


_breakers = {
    "reddit":    CircuitBreaker("reddit"),
    "exa":       CircuitBreaker("exa"),
    "youtube":   CircuitBreaker("youtube"),
    "foreplay":  CircuitBreaker("foreplay"),
    "twitter":   CircuitBreaker("twitter"),
    "tiktok":    CircuitBreaker("tiktok"),
    "instagram": CircuitBreaker("instagram"),
    "threads":   CircuitBreaker("threads"),
    "hn":        CircuitBreaker("hn"),
    "github":    CircuitBreaker("github"),
}


def get_breaker(name: str) -> CircuitBreaker:
    return _breakers[name]


def all_statuses() -> list[dict]:
    return [b.status() for b in _breakers.values()]
