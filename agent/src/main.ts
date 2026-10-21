#!/usr/bin/env node
// Container entrypoint: one night, then exit. The platform injects
// AUTOTUNE_API_URL / AUTOTUNE_API_KEY / AUTOTUNE_SESSION_ID; the LLM endpoint
// comes from the policy's registered env (LLM_BASE_URL, LLM_MODEL, and
// LLM_API_KEY as a secret variable) or the campaign's policy params.

import { SessionOver } from "./contract/client.js";
import { runNight } from "./night.js";

try {
  process.exitCode = await runNight();
} catch (err) {
  if (err instanceof SessionOver) {
    console.log("the session is over; exiting");
    process.exitCode = 0;
  } else {
    console.error(err);
    process.exitCode = 1;
  }
}
