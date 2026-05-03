"""Async HTTP client for the Tap Electric API.

All read endpoints verified 2026-04-22 against the Reference docs.
Write endpoints (OCPP passthrough + Reset) are the final unverified
surface — they exist and 200 on the test-request button, but real
behaviour on a physical charger is untested from this client.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp import ClientResponseError, ClientTimeout

from .const import (
    API_VERSION,
    AUTH_HEADER_APIKEY,
    AUTH_HEADER_BEARER,
    AUTH_HEADER_TAP,
    AUTH_SCHEME,
    CHARGING_LIMIT_DEFAULT_A,
    CHARGING_LIMIT_OFF_A,
    DEFAULT_BASE_URL,
    METER_DATA_LIMIT,
    PATH_CHARGER_DETAIL,
    PATH_CHARGER_OCPP_GET,
    PATH_CHARGER_OCPP_SEND,
    PATH_CHARGER_RESET,
    PATH_CHARGER_SESSIONS,
    PATH_CHARGERS_LIST,
    PATH_LOCATIONS_LIST,
    PATH_METER_DATA_PUSH,
    PATH_SESSION_METER_DATA,
    PATH_TARIFFS,
    PATH_CHARGER_UNLOCK,       # [ADDED]
    PATH_CHARGER_REMOTE_START, # [ADDED]
)
from .ocpp import reset as ocpp_reset
from .ocpp import set_charging_profile, build_ocpp_request

_LOGGER = logging.getLogger(__name__)
_TIMEOUT = ClientTimeout(total=20)


class TapElectricError(Exception):
    """Base exception for Tap Electric API."""


class TapElectricAuthError(TapElectricError):
    """401 / 403 — API key missing or invalid."""


class TapElectricNotFoundError(TapElectricError):
    """404 — resource not found."""


class TapElectricServerError(TapElectricError):
    """5xx — server side problem, retry may help."""


class TapElectricClient:
    """Thin async wrapper around the Tap Electric REST API."""

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession,
        base_url: str = DEFAULT_BASE_URL,
        auth_scheme: str = AUTH_SCHEME,
    ) -> None:
        self._api_key = api_key
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._auth_scheme = auth_scheme

    # ── internal helpers ────────────────────────────────────────────────

    def _url(self, path: str, **fmt: Any) -> str:
        path = path.format(**fmt) if fmt else path
        return f"{self._base_url}/api/{API_VERSION}{path}"

    def _auth_headers(self) -> dict[str, str]:
        if self._auth_scheme == "x-api-key":
            return {AUTH_HEADER_APIKEY: self._api_key}
        if self._auth_scheme == "bearer":
            return {AUTH_HEADER_BEARER: f"Bearer {self._api_key}"}
        if self._auth_scheme == "x-tap-api-key":
            return {AUTH_HEADER_TAP: self._api_key}
        if self._auth_scheme == "basic":
            return {AUTH_HEADER_BEARER: f"Basic {self._api_key.removeprefix('sk_')}"}
        raise TapElectricError(f"Unknown auth_scheme: {self._auth_scheme}")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        **fmt: Any,
    ) -> Any:
        url = self._url(path, **fmt)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **self._auth_headers(),
        }
        _LOGGER.debug("%s %s params=%s body=%s", method, url, params, json)
        try:
            async with self._session.request(
                method, url,
                headers=headers, json=json, params=params, timeout=_TIMEOUT,
            ) as resp:
                text = await resp.text()
                if resp.status in (401, 403):
                    raise TapElectricAuthError(
                        f"Auth failed ({resp.status}): {text[:200]}"
                    )
                if resp.status == 404:
                    raise TapElectricNotFoundError(f"Not found: {url}")
                if resp.status >= 500:
                    raise TapElectricServerError(
                        f"Server error {resp.status}: {text[:200]}"
                    )
                if resp.status >= 400:
                    raise TapElectricError(
                        f"HTTP {resp.status} on {method} {url}: {text[:200]}"
                    )
                if not text:
                    return None
                return await resp.json(content_type=None)
        except asyncio.TimeoutError as err:
            raise TapElectricServerError(f"Timeout calling {url}") from err
        except ClientResponseError as err:
            raise TapElectricError(f"Transport error: {err}") from err

    # ── Chargers (verified) ─────────────────────────────────────────────

    async def list_chargers(self) -> list[dict]:
        data = await self._request("GET", PATH_CHARGERS_LIST)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        return data or []

    async def get_charger(self, charger_id: str) -> dict:
        return await self._request(
            "GET", PATH_CHARGER_DETAIL, charger_id=charger_id
        )

    # ── Charger sessions (verified, replaces /sessions) ─────────────────

    async def list_charger_sessions(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        updated_since: str | None = None,   # ISO8601
    ) -> list[dict]:
        """All charger-sessions visible to this API key.

        Schema per item (verified):
          {id, location:{id}, charger:{id, connectorId},
           wh, startedAt, endedAt|null, updatedAt}
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if updated_since:
            params["updatedSince"] = updated_since
        data = await self._request("GET", PATH_CHARGER_SESSIONS, params=params)
        return data or []

    async def session_meter_data(
        self,
        session_id: str,
        *,
        limit: int = METER_DATA_LIMIT,
        offset: int = 0,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict]:
        """
