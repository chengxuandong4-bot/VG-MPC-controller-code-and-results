#!/usr/bin/env python3
"""Credential-safe requests transport for DeepSeek chat completions."""
from __future__ import annotations

import json
import re
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL.*",
)

import requests


ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
RETRYABLE_HTTP_STATUSES = {429}


def redact_sensitive(value: Any, secret: str = "", limit: int = 2000) -> str:
    """Return diagnostic text with credentials removed."""
    text = str(value or "")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;\"}\]]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)\bbearer\s+[-A-Za-z0-9._~+/]+=*", "Bearer [REDACTED]", text
    )
    text = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]{6,}", "[REDACTED]", text)
    text = re.sub(
        r"(?i)([\"']?(?:api[_-]?key|apikey|access[_-]?token|token)[\"']?\s*[:=]\s*[\"']?)[^\"'\s,&}]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)([?&](?:api[_-]?key|apikey|access[_-]?token|token)=)[^&#\s]+",
        r"\1[REDACTED]",
        text,
    )
    return text[:limit]


def _safe_json(response: Any) -> tuple[Optional[Any], bool]:
    try:
        return response.json(), True
    except Exception:
        return None, False


def _exception_chain(exc: BaseException):
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _transport_exception_category(exc: BaseException) -> str:
    if isinstance(exc, requests.exceptions.Timeout):
        return "timeout"
    if isinstance(exc, requests.exceptions.ConnectionError):
        chain = list(_exception_chain(exc))
        if any(isinstance(item, ConnectionResetError) for item in chain):
            return "connection_reset"
        text = " ".join(str(item).lower() for item in chain)
        if "connection reset" in text or "reset by peer" in text:
            return "connection_reset"
    return "transport_non_retryable"


def _http_retry_category(status: Optional[int]) -> str:
    if status == 429:
        return "http_429"
    if isinstance(status, int) and 500 <= status <= 599:
        return "http_5xx"
    return "http_non_retryable"


def _blank_result() -> Dict[str, Any]:
    return {
        "request_completed": False,
        "http_response_received": False,
        "http_status": None,
        "reason": "",
        "response_content_type": "",
        "provider_response_json_valid": False,
        "assistant_content": "",
        "assistant_content_json": None,
        "provider_error_type": "",
        "provider_error_code": "",
        "provider_message_redacted": "",
        "transport_exception_class": "",
        "transport_exception_message_redacted": "",
        "request_duration_s": 0.0,
        "physical_request_count": 0,
        "retry_count": 0,
        "retry_reasons": [],
        "token_usage": {},
        "provider_request_id": "",
        "http_success": False,
    }


def _numeric_usage(value: Any) -> Any:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): filtered for key, item in value.items() if (filtered := _numeric_usage(item)) is not None}
    return None


def deepseek_chat_request(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    connect_timeout: float,
    read_timeout: float,
    response_format: Dict[str, str],
    allow_transport_retry: bool,
    post: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Send one logical request, with at most one narrowly-scoped retry."""
    if post is None:
        post = requests.post
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": response_format,
    }
    started = time.perf_counter()
    physical_count = 0
    retry_reasons: List[str] = []
    final = _blank_result()

    for attempt in range(2):
        physical_count += 1
        try:
            response = post(
                ENDPOINT,
                headers={
                    "Authorization": "Bearer " + api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=(connect_timeout, read_timeout),
                verify=True,
                allow_redirects=True,
            )
            envelope, envelope_valid = _safe_json(response)
            status = getattr(response, "status_code", None)
            retry_category = _http_retry_category(status)
            retryable = retry_category in {"http_429", "http_5xx"}

            if attempt == 0 and allow_transport_retry and retryable:
                retry_reasons.append(retry_category)
                continue

            content = ""
            usage: Any = {}
            provider_type = ""
            provider_code = ""
            provider_message = ""
            envelope_id = ""
            if isinstance(envelope, dict):
                envelope_id = redact_sensitive(envelope.get("id", ""), api_key, 500)
                usage = _numeric_usage(envelope.get("usage", {})) or {}
                choices = envelope.get("choices", [])
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        content = redact_sensitive(message["content"], api_key, 100000)
                error = envelope.get("error", {})
                if isinstance(error, dict):
                    provider_type = redact_sensitive(error.get("type", ""), api_key)
                    provider_code = redact_sensitive(error.get("code", ""), api_key)
                    provider_message = redact_sensitive(error.get("message", ""), api_key)
            assistant_json = None
            if content:
                try:
                    assistant_json = json.loads(content)
                except Exception:
                    pass
            headers = getattr(response, "headers", {}) or {}
            request_id = ""
            for name in ("x-request-id", "x-ds-request-id", "request-id"):
                if headers.get(name):
                    request_id = redact_sensitive(headers.get(name), api_key, 500)
                    break
            final.update(
                {
                    "request_completed": True,
                    "http_response_received": True,
                    "http_status": status,
                    "reason": redact_sensitive(getattr(response, "reason", ""), api_key),
                    "response_content_type": redact_sensitive(headers.get("Content-Type", ""), api_key),
                    "provider_response_json_valid": envelope_valid,
                    "assistant_content": content,
                    "assistant_content_json": assistant_json,
                    "provider_error_type": provider_type,
                    "provider_error_code": provider_code,
                    "provider_message_redacted": provider_message,
                    "token_usage": usage if isinstance(usage, dict) else {},
                    "provider_request_id": request_id or envelope_id,
                    "http_success": isinstance(status, int) and 200 <= status <= 299,
                }
            )
            break
        except Exception as exc:
            category = _transport_exception_category(exc)
            if (
                attempt == 0
                and allow_transport_retry
                and category in {"timeout", "connection_reset"}
            ):
                retry_reasons.append(category)
                continue
            final.update(
                {
                    "transport_exception_class": type(exc).__name__,
                    "transport_exception_message_redacted": redact_sensitive(exc, api_key),
                }
            )
            break

    final["request_duration_s"] = time.perf_counter() - started
    final["physical_request_count"] = physical_count
    final["retry_count"] = max(0, physical_count - 1)
    final["retry_reasons"] = retry_reasons
    return final


# A short alias keeps call sites readable while preserving a descriptive public API.
request_chat_completion = deepseek_chat_request
