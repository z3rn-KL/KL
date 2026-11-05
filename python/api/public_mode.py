"""
Settings for exposing the demo on the internet. Active only when APP_PASSWORD
is set or PUBLIC_NO_PASSWORD=1; a local run without them behaves as before.

- With APP_PASSWORD, every request needs it (HTTP Basic, any user name).
  PUBLIC_NO_PASSWORD=1 lets anyone with the address in; a password wins if
  both are set.
- Coordinates in JSON responses are rounded to about 110 m, because the data
  are real delivery locations.
- The built frontend is served from the same address as the API.
"""

import base64
import binascii
import json
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from api.config import ANONYMIZE_DELIVERIES, PROJECT_ROOT


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

COORDINATE_DECIMALS = 3

# Ho Chi Minh City and surroundings. Rounding by value rather than by key also
# covers route geometry, which is a list of unnamed [lng, lat] pairs.
LATITUDE_RANGE = (10.0, 11.2)
LONGITUDE_RANGE = (106.0, 107.2)


def _round_coordinate(value: float) -> float:
    in_range = (
        LATITUDE_RANGE[0] <= value <= LATITUDE_RANGE[1]
        or LONGITUDE_RANGE[0] <= value <= LONGITUDE_RANGE[1]
    )

    return round(value, COORDINATE_DECIMALS) if in_range else value


def mask_coordinates(payload):
    if isinstance(payload, float):
        return _round_coordinate(payload)

    if isinstance(payload, list):
        return [mask_coordinates(item) for item in payload]

    if isinstance(payload, dict):
        return {key: mask_coordinates(item) for key, item in payload.items()}

    return payload


def password_matches(header: str | None, password: str) -> bool:
    if not header or not header.lower().startswith("basic "):
        return False

    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return False

    _, separator, supplied = decoded.partition(":")

    return bool(separator) and secrets.compare_digest(
        supplied.encode("utf-8"),
        password.encode("utf-8"),
    )


def frontend_file(path: str, dist: Path) -> Path | None:
    """
    The requested file inside ``dist``, or index.html for a client-side route.
    """

    root = dist.resolve()
    candidate = (root / path.lstrip("/")).resolve()

    if candidate.is_file() and candidate.is_relative_to(root):
        return candidate

    index = root / "index.html"

    return index if index.is_file() else None


def install_public_mode(
    app: FastAPI,
    password: str | None = None,
    dist: Path = FRONTEND_DIST,
    open_access: bool | None = None,
) -> bool:
    """
    Add the public-mode middleware when a password or open access is
    configured. Returns whether it was installed.
    """

    password = password if password is not None else os.getenv("APP_PASSWORD")

    if open_access is None:
        open_access = os.getenv("PUBLIC_NO_PASSWORD") == "1"

    if not password and not open_access:
        return False

    if not ANONYMIZE_DELIVERIES:
        raise RuntimeError(
            "Public mode is on but XEDU_ANONYMIZE=0: refusing to expose "
            "non-anonymized deliveries."
        )

    @app.middleware("http")
    async def public_mode(request: Request, call_next):
        if password and not password_matches(
            request.headers.get("authorization"),
            password,
        ):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="NEXUS"'},
            )

        path = request.url.path

        if not path.startswith("/api/"):
            target = frontend_file(path, dist)

            if target is None:
                return JSONResponse(
                    {"detail": "Frontend has not been built."},
                    status_code=503,
                )

            return FileResponse(target)

        response = await call_next(request)

        if "application/json" not in response.headers.get("content-type", ""):
            return response

        body = b"".join([chunk async for chunk in response.body_iterator])

        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() != "content-length"
        }

        return Response(
            content=json.dumps(
                mask_coordinates(json.loads(body)),
                ensure_ascii=False,
            ),
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
        )

    return True
