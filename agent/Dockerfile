# The agent policy: Node, the pi agent runtime, and our harness. Two stages:
# build the TypeScript, then ship only the production dependencies, the
# compiled harness, the prompts and the third-party notices.
#
# Point BASE at whatever Node your registry mirrors, and NPM_REGISTRY at an npm
# mirror if the build cannot reach registry.npmjs.org.
ARG BASE=node:22-slim
FROM ${BASE} AS build
ARG NPM_REGISTRY=https://registry.npmjs.org/
WORKDIR /agent
COPY package.json package-lock.json ./
RUN npm ci --registry="${NPM_REGISTRY}"
COPY tsconfig.json ./
COPY src/ ./src/
COPY test/ ./test/
COPY scripts/ ./scripts/
RUN npx tsc -p tsconfig.json && node scripts/licenses.mjs

FROM ${BASE}
ARG NPM_REGISTRY=https://registry.npmjs.org/
WORKDIR /agent
ENV NODE_ENV=production
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --registry="${NPM_REGISTRY}" && npm cache clean --force
COPY --from=build /agent/dist/src/ ./dist/src/
COPY --from=build /agent/THIRD_PARTY_LICENSES ./
COPY prompts/ ./prompts/
COPY LICENSE ./

# The platform injects AUTOTUNE_API_URL / AUTOTUNE_API_KEY / AUTOTUNE_SESSION_ID;
# LLM_BASE_URL / LLM_MODEL / LLM_API_KEY come from the policy's registered env.
ENTRYPOINT ["node", "dist/src/main.js"]
