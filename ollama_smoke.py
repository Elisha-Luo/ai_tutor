"""One real Ollama request through the same bounded client as the web app.

Run from this directory with ``python ollama_smoke.py``.  Does not import app,
open chat.db, or print the API key.  Prints only the real model reply.
"""

import os
import sys
from pathlib import Path

import rag
from env_utils import load_dotenv
from model_client import BoundedModelClient
from provider_config import OLLAMA_BASE_URL, OLLAMA_MODEL


def main():
    load_dotenv(str(Path(__file__).with_name(".env")))
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not key:
        raise SystemExit("OLLAMA_API_KEY 未设置；不会尝试其他服务商的密钥。")

    client = BoundedModelClient(api_key=key, base_url=OLLAMA_BASE_URL, timeout=30)
    result, diagnostic = rag.generate_answer_with_context_and_profile(
        "What is the past tense of go? Explain briefly.",
        [], [], None, client, OLLAMA_MODEL,
    )
    if diagnostic != rag.DIAG_OK or result.get("decision") != rag.DECISION_GENERAL_ANSWER:
        raise SystemExit(
            "Ollama 请求结果未通过应用验收："
            f"diagnostic={diagnostic}, decision={result.get('decision')}, "
            f"provider_error={client.last_error_code}"
        )
    answer = result["answer"]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(answer.strip())


if __name__ == "__main__":
    main()
