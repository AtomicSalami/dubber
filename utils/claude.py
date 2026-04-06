"""
utils/claude.py — Thin wrapper around the Anthropic Messages API.
Intentionally keeps no state so it can be imported anywhere.
"""

import json
import urllib.request
import urllib.error

from config import CLAUDE_MODEL


def claude_call(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4096,
) -> str:
    """
    Send a single-turn message to Claude and return the response text.

    Raises:
        RuntimeError: on any HTTP error from the API.
    """
    payload = json.dumps({
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Claude API error {e.code}: {e.read().decode()}") from e


def parse_json_response(raw: str) -> list | dict:
    """
    Parse Claude's response as JSON, stripping any markdown code fences
    (```json ... ```) that the model may have added despite being told not to.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        # parts[1] is the content between the first pair of fences
        raw = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())