"""Container entrypoint. This is the whole policy:

    run_policy(SearchLoop(RandomSearch()))

Copy this directory, swap ``RandomSearch`` for your own ``Strategy``, and you have a
new policy. Everything else — the contract, the night, the clock — is the SDK's.
"""

from __future__ import annotations

import logging

from autotune_policy import SearchLoop, __version__, run_policy
from strategies.random_search import RandomSearch


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return run_policy(SearchLoop(RandomSearch(), sdk_version=__version__))


if __name__ == "__main__":
    raise SystemExit(main())
