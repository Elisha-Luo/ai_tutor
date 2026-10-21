"""Run one model request with a wall-clock deadline, including keep-alive waits.

The worker imports neither app nor its database/configuration loader. Credentials
and messages travel through an anonymous stdin pipe, never command-line args or
temporary files. Its only stdout is the result protocol; stderr is discarded.
"""

import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

_FAILURE_CODES = frozenset({
    "timeout", "connection_error", "authentication_error", "rate_limited",
    "upstream_error", "response_error",
})

class ModelRequestError(Exception):
    """A fixed, content-free error safe for the caller to classify."""


class ModelRequestTimeout(ModelRequestError):
    pass


class BoundedModelClient:
    """Small adapter for the chat.completions.create interface used by rag."""

    def __init__(self, *, api_key, base_url, timeout=30):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self.last_error_code = None
        self.chat = self
        self.completions = self

    def with_timeout(self, timeout):
        """Use a shorter deadline for a follow-up call; do not mutate the first client."""
        return BoundedModelClient(api_key=self._api_key, base_url=self._base_url,
                                  timeout=timeout)

    def create(self, *, model, messages):
        self.last_error_code = None
        payload = json.dumps({
            "api_key": self._api_key, "base_url": self._base_url,
            "timeout": self._timeout, "model": model, "messages": messages,
        }, ensure_ascii=False).encode("utf-8")
        worker_env = os.environ.copy()
        for name in ("OLLAMA_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENAI_LOG",
                     "FLASK_SECRET_KEY", "INVITE_CODE_PEPPER",
                     "ADMIN_MINT_TOKEN", "CHAT_DB_PATH"):
            worker_env.pop(name, None)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        started = time.monotonic()
        with subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=worker_env, creationflags=flags,
        ) as process:
            try:
                output, _ = process.communicate(
                    input=payload,
                    timeout=max(0.001, self._timeout - (time.monotonic() - started)),
                )
            except subprocess.TimeoutExpired:
                # Kill/reap before returning: releasing the app lock must not
                # leave the original request running in a background thread.
                process.kill()
                process.communicate()
                self.last_error_code = "timeout"
                raise ModelRequestTimeout("model request deadline exceeded") from None
            except BaseException:
                process.kill()
                process.communicate()
                raise
            if process.returncode != 0:
                self.last_error_code = "timeout" if process.returncode == 124 else "response_error"
                raise ModelRequestError("model worker failed")
        try:
            result = json.loads(output)
            if result.get("ok") is not True or not isinstance(result.get("content"), str):
                code = result.get("error")
                self.last_error_code = code if code in _FAILURE_CODES else "response_error"
                raise ValueError
            content = result["content"]
        except (ValueError, TypeError, AttributeError):
            if self.last_error_code is None:
                self.last_error_code = "response_error"
            raise ModelRequestError("model service unavailable") from None
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content))])


def _worker():
    # Even if a parent has enabled verbose SDK logging, never expose request
    # headers, prompts, responses or exception text through worker logs.
    logging.disable(logging.CRITICAL)
    watchdog = None
    try:
        payload = json.loads(sys.stdin.buffer.read())
        # Also bound an orphaned worker if its web-process parent exits early.
        watchdog = threading.Timer(payload["timeout"], lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        from openai import OpenAI

        with OpenAI(api_key=payload["api_key"], base_url=payload["base_url"],
                    timeout=payload["timeout"], max_retries=0) as client:
            response = client.chat.completions.create(
                model=payload["model"], messages=payload["messages"],
                response_format={"type": "json_object"})
            content = response.choices[0].message.content
        result = {"ok": True, "content": content}
    except Exception as exc:
        # Only a fixed category crosses the pipe; no exception text or headers.
        code = {
            "APITimeoutError": "timeout", "APIConnectionError": "connection_error",
            "AuthenticationError": "authentication_error", "RateLimitError": "rate_limited",
            "InternalServerError": "upstream_error", "APIStatusError": "upstream_error",
        }.get(type(exc).__name__, "response_error")
        result = {"ok": False, "error": code}
    finally:
        if watchdog is not None:
            watchdog.cancel()
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    _worker()
