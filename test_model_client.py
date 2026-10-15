"""Real worker + local fake HTTP server. No provider calls or real credentials."""

import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import model_client


class TestBoundedModelClient(unittest.TestCase):
    def setUp(self):
        self.mode = "success"
        self.calls = 0
        self.disconnected = threading.Event()
        self.stop = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                owner.calls += 1
                if owner.mode == "error":
                    body = b'{"error":{"message":"private-error-marker","type":"server_error"}}'
                    self.send_response(503)
                else:
                    body = json.dumps({
                        "id": "test", "object": "chat.completion", "created": 1,
                        "model": "fake-model", "choices": [{"index": 0,
                            "finish_reason": "stop", "message": {
                                "role": "assistant", "content": "fake-answer"}}],
                    }).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                if owner.mode in ("success", "error"):
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if owner.mode in ("success", "error"):
                    self.wfile.write(body)
                    return
                while not owner.stop.wait(0.05):
                    if owner.mode == "heartbeat":
                        try:
                            self.wfile.write(b"\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            owner.disconnected.set()
                            return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env_patch = patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        self.env_patch.start()

    def tearDown(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.env_patch.stop()

    def client(self, timeout=10):
        return model_client.BoundedModelClient(
            api_key="offline-test-key", base_url=f"http://127.0.0.1:{self.server.server_port}/v1",
            timeout=timeout)

    def call(self, client):
        return client.chat.completions.create(
            model="fake-model", messages=[{"role": "user", "content": "offline-test-question"}])

    def test_real_worker_returns_the_completion_without_importing_app(self):
        response = self.call(self.client())
        self.assertEqual(response.choices[0].message.content, "fake-answer")
        self.assertEqual(self.calls, 1)

    def test_provider_503_is_not_retried_and_error_contents_are_hidden(self):
        self.mode = "error"
        client = self.client()
        with self.assertRaises(model_client.ModelRequestError) as caught:
            self.call(client)
        self.assertEqual(self.calls, 1)
        self.assertEqual(client.last_error_code, "upstream_error")
        for marker in ("private-error-marker", "offline-test-key", "offline-test-question"):
            self.assertNotIn(marker, str(caught.exception))

    def test_silent_response_cannot_hold_the_worker_forever(self):
        self.mode = "silent"
        started = time.monotonic()
        with self.assertRaises(model_client.ModelRequestError):
            self.call(self.client(timeout=8))
        self.assertLess(time.monotonic() - started, 12)
        self.assertEqual(self.calls, 1)

    def test_continuous_empty_lines_still_hit_deadline_and_worker_is_reaped(self):
        self.mode = "heartbeat"
        processes = []
        original = model_client.subprocess.Popen

        def record(*args, **kwargs):
            self.assertNotIn("offline-test-key", str(args))
            self.assertNotIn("offline-test-question", str(args))
            self.assertNotIn("CHAT_DB_PATH", kwargs["env"])
            self.assertNotIn("ADMIN_MINT_TOKEN", kwargs["env"])
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        started = time.monotonic()
        with patch.object(model_client.subprocess, "Popen", side_effect=record):
            with self.assertRaises(model_client.ModelRequestError):
                self.call(self.client(timeout=8))
        self.assertLess(time.monotonic() - started, 12)
        self.assertEqual(self.calls, 1)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll(), "request worker survived deadline")
        self.assertTrue(self.disconnected.wait(2), "upstream socket was not closed")
        # No old request remains; the same adapter can complete the next call.
        self.mode = "success"
        self.assertEqual(self.call(self.client()).choices[0].message.content, "fake-answer")


if __name__ == "__main__":
    unittest.main()
