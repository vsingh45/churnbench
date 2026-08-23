"""Thin httpx client for the mock SaaS API.

The mock SaaS adds SAAS_LATENCY_MS (default 350 ms) to every request and
enforces SAAS_RATE_LIMIT_PER_MIN (default 60 req/min) with HTTP 429 responses.
This client handles 429s with exponential backoff (max 3 retries) so callers
see the real waiting time in their latency measurements.
"""

from __future__ import annotations

import time
from typing import Any

import httpx


class SaasClient:
    def __init__(self, base_url: str = "http://localhost:8010") -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=30.0)

    # ── Typed convenience methods ─────────────────────────────────────────────

    def get_entitlements(self, user_ext_id: str) -> dict[str, Any]:
        return self.raw_get(f"/entitlements/{user_ext_id}")  # type: ignore[no-any-return]

    def get_tickets(
        self,
        status: str | None = None,
        product_sku: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if status:
            params["status"] = status
        if product_sku:
            params["product_sku"] = product_sku
        return self._get("/tickets", params=params)  # type: ignore[no-any-return]

    def get_product_utilization(self, product_sku: str) -> dict[str, Any]:
        return self.raw_get(f"/products/{product_sku}/current-utilization")  # type: ignore[no-any-return]

    def health(self) -> bool:
        try:
            r = self._http.get(self._base_url + "/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    # ── Raw GET — used by the naive arm to give the LLM free-form access ──────

    def raw_get(self, path: str) -> Any:
        """GET any path with backoff on 429."""
        return self._get(path)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(
        self,
        path: str,
        params: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> Any:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                r = self._http.get(self._base_url + path, params=params)
                if r.status_code == 429:
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= 2.0
                        continue
                    r.raise_for_status()
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2.0
        raise RuntimeError(f"SaaS GET {path} failed after {max_retries + 1} attempts") from last_exc

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SaasClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
