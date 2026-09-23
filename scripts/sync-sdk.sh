#!/usr/bin/env bash
# Copy the SDK out to every policy directory.
#
# Policies vendor autotune_policy/ rather than depending on it: it is stdlib-only
# so that a policy image needs no wheels and no reachable package index, which is
# the difference between "works" and "does not work" on an air-gapped GPU host.
# The price is copies, and this script plus the `sdk-copies-match` CI job is how
# they are kept honest.
#
# Run it after changing autotune_policy/, then commit the result.
set -euo pipefail

cd "$(dirname "$0")/.."

for policy in */; do
  policy=${policy%/}
  [ "$policy" = autotune_policy ] && continue
  [ -f "$policy/Dockerfile" ] || continue
  rsync -a --delete --exclude='__pycache__' autotune_policy/ "$policy/autotune_policy/"
  echo "synced -> $policy/autotune_policy/"
done
