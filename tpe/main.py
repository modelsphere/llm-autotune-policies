"""Container entrypoint: the TPE strategy in the SDK's standard night.

    run_policy(SearchLoop(TPE()))

``TPE_MAX_TRIALS`` (default 60) caps a night's trials and
``TPE_STARTUP_TRIALS`` (default 6) sets how many seeded random draws come
before the model.
"""

from __future__ import annotations

import logging
import os

from autotune_policy import SearchLoop, __version__, run_policy
from strategies.tpe import TPE


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    strategy = TPE(
        max_trials=int(os.environ.get("TPE_MAX_TRIALS", "60")),
        n_startup_trials=int(os.environ.get("TPE_STARTUP_TRIALS", "6")),
    )
    return run_policy(SearchLoop(strategy, sdk_version=__version__))


if __name__ == "__main__":
    raise SystemExit(main())
