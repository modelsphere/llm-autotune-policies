"""autotune-policy — the SDK a policy container uses to run a night of tuning.

The whole contract in one line:

    from autotune_policy import run_policy, SearchLoop
    from strategies.random_search import RandomSearch

    run_policy(SearchLoop(RandomSearch()))

``run_policy`` reads the boot env, ``SearchLoop`` drives the contract, and your
``Strategy`` decides what to tune. See docs/api/policy-contract.md (platform
repo) for the wire protocol; nothing here needs it open.
"""

from .client import (
    ConfigRejected,
    NoCapacity,
    PhaseError,
    PolicyClient,
    PolicyConfig,
    PolicyError,
    SessionOver,
)
from .loop import SearchLoop
from .run import run_policy
from .strategy import Strategy, TrialResult, build_contender

__version__ = "0.1.0"

__all__ = [
    "ConfigRejected",
    "NoCapacity",
    "PhaseError",
    "PolicyClient",
    "PolicyConfig",
    "PolicyError",
    "SearchLoop",
    "SessionOver",
    "Strategy",
    "TrialResult",
    "__version__",
    "build_contender",
    "run_policy",
]
