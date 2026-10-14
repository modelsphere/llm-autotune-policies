# A policy image is tiny: the SDK is stdlib-only, so there is nothing to pip
# install and no wheels to pull on a restricted network. Point BASE at whatever
# Python your registry mirrors.
ARG BASE=python:3.12-slim
FROM ${BASE}

WORKDIR /policy
ENV AUTOTUNE_LOG_LEVEL=INFO
COPY autotune_policy/ ./autotune_policy/
COPY strategies/ ./strategies/
COPY main.py dry_run.py decision_log.py ./
COPY examples/dry_run_manifest.json ./examples/dry_run_manifest.json

# The platform injects AUTOTUNE_API_URL / AUTOTUNE_API_KEY / AUTOTUNE_SESSION_ID
# at launch; the policy dials out from there.
ENTRYPOINT ["python", "-u", "main.py"]
