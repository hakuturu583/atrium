"""In-sandbox A2A↔tabbyAPI bridge (container-side).

Importing ``bridge.server`` pulls in the container-only HTTP/ASGI stack
(``httpx``, ``starlette``, ``uvicorn``, ``a2a-sdk[http-server]``). Do this only
inside the sandbox image — never from the Atrium host package.
"""

from __future__ import annotations

__all__: list[str] = []
