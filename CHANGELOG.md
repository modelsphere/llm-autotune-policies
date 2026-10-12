# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The SDK's search loop logs each step at INFO: the start, the plan, every
  launch and its benchmark result, the platform's commands, and the exit. The
  platform saves a policy container's log when the session ends; a policy that
  logged nothing left an empty one.
- Policy images set `PYTHONUNBUFFERED=1`, so nothing printed is lost in a buffer
  when the platform removes the container.

## [0.1.1] - 2026-10-10

### Added
- `random-search` is published on a version tag as
  `ghcr.io/modelsphere/llm-autotune-policy-random-search:<version>`, for
  amd64 and arm64, so it can be registered without building it. The CI checks
  that the tag matches the policy's version.
- README: how to register a policy image through the platform's API
  (`POST /api/policies`) (#1).
- CI gate for the license text and committed credentials.
- Dependabot for each policy's dev tools (uv) and for the GitHub Actions.
- `NOTICE` and this changelog.

### Changed
- GitHub Actions pinned to commit SHAs; the workflow token is read-only, and a
  new push to a pull request cancels the run for the previous one.

### Removed
- The CI step that grepped for a list of internal host names.

## [0.1.0] - 2026-09-24

First release.

### Added
- `autotune_policy`, the policy SDK: stdlib only, and vendored verbatim into
  each policy so an image builds with no package index.
  `run_policy(SearchLoop(strategy))` implements the policy contract.
- `random-search`: the reference policy and the fork-and-edit template.
- `chaos`: a fault-injection policy that emits common policy failures, chosen
  with `CHAOS_SCENARIO`, to test how the platform handles them.
- `scripts/sync-sdk.sh` to refresh the vendored SDK copies.
- CI: tests and `ruff` for both policies, a check that the vendored SDK copies
  match the copy of record, and an image build for each policy. No image is
  published; the README explains how to build one and get it onto a machine
  without a registry.

[Unreleased]: https://github.com/modelsphere/llm-autotune-policies/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/modelsphere/llm-autotune-policies/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/modelsphere/llm-autotune-policies/releases/tag/v0.1.0
