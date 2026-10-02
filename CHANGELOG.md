# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/modelsphere/llm-autotune-policies/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/modelsphere/llm-autotune-policies/releases/tag/v0.1.0
