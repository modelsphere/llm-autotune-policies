"""Chaos policy entrypoint.

Not a tuner — a fault injector for testing how the platform handles a policy
that misbehaves. The campaign chooses the fault with the ``CHAOS_SCENARIO``
environment variable (default ``happy``); see README.md for the menu and the
platform behaviour each one should produce.

    CHAOS_SCENARIO=idle python main.py
"""

from __future__ import annotations

import logging
import os
import sys

from autotune_policy import PolicyClient, PolicyConfig, SessionOver
from faults import runner_for, scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s chaos %(message)s")
log = logging.getLogger("chaos")


def main() -> int:
    scenario = os.environ.get("CHAOS_SCENARIO", "happy")
    log.info("scenario=%s (menu: %s)", scenario, ", ".join(scenarios()))
    client = PolicyClient(PolicyConfig.from_env())
    runner = runner_for(scenario)
    try:
        return runner(client)
    except SessionOver as exc:
        log.info("session ended for us (%s) — clean exit", exc)
        return 0


if __name__ == "__main__":
    sys.exit(main())
