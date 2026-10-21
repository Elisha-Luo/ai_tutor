"""Public model endpoint settings shared by the web app and its smoke test.

The API key is deliberately not defined here; callers must read OLLAMA_API_KEY.
"""

OLLAMA_BASE_URL = "https://ollama.com/v1"
OLLAMA_MODEL = "glm-5.3-flash"
