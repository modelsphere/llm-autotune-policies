# A policy image is tiny: the SDK is stdlib-only, so there is nothing to pip
# install and no wheels to pull on a restricted network. Point BASE at whatever
# Python your registry mirrors.
ARG BASE=python:3.12-slim
FROM ${BASE}

WORKDIR /policy
COPY autotune_policy/ ./autotune_policy/
COPY faults.py main.py ./

# The platform injects AUTOTUNE_API_URL / AUTOTUNE_API_KEY / AUTOTUNE_SESSION_ID
# at launch; the policy dials out from there. Pick the fault with CHAOS_SCENARIO
# (default "happy"); see README.md.
ENV CHAOS_SCENARIO=happy
ENTRYPOINT ["python", "main.py"]
