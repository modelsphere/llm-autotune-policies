"""``run_policy`` — the single call a policy's ``main`` makes.

Reads the three boot env vars into a client, runs the loop, and turns the
platform's "the session is over for you" (401/410) into a clean exit — because
that is exactly what it means.
"""

from __future__ import annotations

import logging

from .client import PolicyClient, PolicyConfig, SessionOver
from .loop import SearchLoop

logger = logging.getLogger("autotune_policy")


def run_policy(loop: SearchLoop, *, config: PolicyConfig | None = None,
               client: PolicyClient | None = None) -> int:
    """Drive ``loop`` to completion. Returns the process exit code (0)."""
    if client is None:
        client = PolicyClient(config or PolicyConfig.from_env())
    try:
        return loop.run(client)
    except SessionOver as exc:
        logger.info("session ended for us (%s); exiting cleanly", exc)
        return 0
