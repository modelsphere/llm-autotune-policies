# Unlike random-search, this policy has dependencies (Optuna and numpy), so the
# image installs them from requirements.txt, which is exported from uv.lock
# with hashes. PIP_INDEX_URL points the install at a mirror on a restricted
# network. Point BASE at whatever Python your registry mirrors.
ARG BASE=python:3.12-slim
FROM ${BASE}

WORKDIR /policy
ENV PYTHONUNBUFFERED=1
ARG PIP_INDEX_URL
COPY requirements.txt ./
RUN pip install --no-cache-dir --require-hashes -r requirements.txt \
        ${PIP_INDEX_URL:+--index-url "$PIP_INDEX_URL"}
COPY autotune_policy/ ./autotune_policy/
COPY strategies/ ./strategies/
COPY main.py ./

# The platform injects AUTOTUNE_API_URL / AUTOTUNE_API_KEY / AUTOTUNE_SESSION_ID
# at launch; the policy dials out from there.
ENTRYPOINT ["python", "main.py"]
