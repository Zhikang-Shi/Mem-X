"""Dependency-free client for OpenAI-compatible VLM endpoints."""

from __future__ import annotations

import fcntl
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


@dataclass(frozen=True)
class VLMCompletion:
    text: str
    model: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None


def _acquire_concurrency_slot() -> TextIO | None:
    """Cross-process cap for live VLM Top3 calls.

    ``SG_VLM_MAX_CONCURRENCY`` slots are file locks under
    ``SG_VLM_CONCURRENCY_LOCK_DIR``.  Unset or 0 disables the gate.
    """

    raw = os.environ.get("SG_VLM_MAX_CONCURRENCY", "").strip()
    if not raw:
        return None
    slots = int(raw)
    if slots <= 0:
        return None
    lock_dir = Path(
        os.environ.get("SG_VLM_CONCURRENCY_LOCK_DIR", "/tmp/sg_vlm_slots")
    )
    lock_dir.mkdir(parents=True, exist_ok=True)
    while True:
        for index in range(slots):
            handle = (lock_dir / f"slot_{index:02d}.lock").open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            return handle
        time.sleep(0.05)


def _release_concurrency_slot(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class OpenAICompatibleVLMClient:
    def __init__(
        self, *, base_url: str, api_key: str, model: str,
        timeout_s: float = 90.0, max_retries: int = 2, json_mode: bool = False,
        user_agent: str = "lifelong-home-nav-memory/1",
        ca_bundle: str | os.PathLike[str] | None = None,
    ) -> None:
        if not str(base_url).strip() or not str(model).strip():
            raise ValueError("base_url and model are required")
        if timeout_s <= 0 or max_retries < 0:
            raise ValueError("invalid timeout or retry count")
        base = str(base_url).rstrip("/")
        self.endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.api_key, self.model = str(api_key or ""), str(model)
        self.timeout_s, self.max_retries = float(timeout_s), int(max_retries)
        self.json_mode, self.user_agent = bool(json_mode), str(user_agent)
        self.ca_bundle = self._resolve_ca_bundle(ca_bundle)
        self.ssl_context = ssl.create_default_context(
            cafile=str(self.ca_bundle) if self.ca_bundle is not None else None
        )

    @staticmethod
    def _resolve_ca_bundle(
        explicit: str | os.PathLike[str] | None,
    ) -> Path | None:
        """Resolve a trusted CA bundle without ever disabling verification.

        Relocatable virtual environments can retain the absolute OpenSSL CA
        path from the machine on which Python was built. In a container that
        path may be absent even though the image has a valid system bundle.
        """

        requested = explicit or os.environ.get("SG_VLM_CA_BUNDLE")
        if requested:
            path = Path(requested).expanduser()
            if not path.is_file():
                raise ValueError(f"VLM CA bundle does not exist: {path}")
            return path.resolve()

        verify_paths = ssl.get_default_verify_paths()
        if verify_paths.cafile and Path(verify_paths.cafile).is_file():
            return None

        candidates = [
            os.environ.get("SSL_CERT_FILE"),
            os.environ.get("REQUESTS_CA_BUNDLE"),
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/pki/tls/certs/ca-bundle.crt",
        ]
        try:
            import certifi

            candidates.append(certifi.where())
        except ImportError:
            pass
        for candidate in candidates:
            if candidate and Path(candidate).expanduser().is_file():
                return Path(candidate).expanduser().resolve()
        raise RuntimeError(
            "no trusted CA bundle is available for the VLM HTTPS endpoint; "
            "set SG_VLM_CA_BUNDLE"
        )

    @classmethod
    def from_env(cls) -> "OpenAICompatibleVLMClient":
        base_url = os.environ.get("BASE_URL") or os.environ.get("SG_VLM_BASE_URL")
        model = os.environ.get("MODEL_NAME") or os.environ.get("SG_VLM_MODEL")
        if not base_url or not model:
            raise ValueError("set BASE_URL and MODEL_NAME")
        return cls(
            base_url=base_url,
            api_key=os.environ.get("API_KEY") or os.environ.get("SG_VLM_API_KEY", ""),
            model=model, timeout_s=float(os.environ.get("SG_VLM_TIMEOUT_S", "90")),
            max_retries=int(os.environ.get("SG_VLM_MAX_RETRIES", "2")),
            json_mode=os.environ.get("SG_VLM_JSON_MODE", "0").lower() in {"1", "true", "yes", "on"},
            ca_bundle=os.environ.get("SG_VLM_CA_BUNDLE"),
        )

    def complete(
        self, messages: Sequence[dict[str, Any]], *, max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> VLMCompletion:
        payload: dict[str, Any] = {
            "model": self.model, "messages": list(messages),
            "max_tokens": int(max_tokens), "temperature": float(temperature),
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        last_error: BaseException | None = None
        slot = _acquire_concurrency_slot()
        try:
            return self._complete_locked(
                payload, headers, last_error
            )
        finally:
            _release_concurrency_slot(slot)

    def _complete_locked(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        last_error: BaseException | None,
    ) -> VLMCompletion:
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                self.endpoint, json.dumps(payload, ensure_ascii=False).encode(), headers, method="POST"
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_s,
                    context=self.ssl_context,
                ) as response:
                    decoded = json.loads(response.read().decode())
                    request_id = response.headers.get("x-request-id")
                choices = decoded.get("choices") or []
                if not choices:
                    raise RuntimeError("VLM response has no choices")
                choice = choices[0]
                content = (choice.get("message") or {}).get("content")
                if isinstance(content, list):
                    content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("VLM response has no text")
                return VLMCompletion(
                    content.strip(), decoded.get("model") or self.model,
                    decoded.get("usage") or {}, request_id or decoded.get("id"),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read(2048).decode(errors="replace")
                last_error = RuntimeError(f"VLM HTTP {exc.code}: {detail[:1000]}")
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(0.5 * 2**attempt)
        raise RuntimeError(f"VLM request failed: {last_error}")
