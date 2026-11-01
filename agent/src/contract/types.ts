// Wire shapes of the policy session API, generated from the platform's OpenAPI
// (contract/policy-api.openapi.json → api.d.ts via `npm run gen:contract`).
// Regenerate when the contract gains fields; nothing here is hand-written
// except the aliases.

import type { components } from "./api.js";

type Schemas = components["schemas"];

export type Manifest = Schemas["SessionManifest"];
export type HeartbeatOut = Schemas["HeartbeatOut"];
export type LaunchOut = Schemas["LaunchOut"];
export type BenchmarkOut = Schemas["BenchmarkOut"];
export type ResultsPart = Schemas["ResultsPart"];
export type EnvironmentPart = Schemas["EnvironmentPart"];
export type ConfigCheckOut = Schemas["ConfigCheckOut"];
export type ContenderIn = Schemas["ContenderIn"];
export type ContenderOut = Schemas["ContenderOut"];
export type ContendersEcho = Schemas["ContendersEcho"];
export type PlanExtra = Schemas["PlanExtra"];
export type TrialOut = Schemas["TrialOut"];
export type VocabularyOut = Schemas["VocabularyOut"];

/** A flat dict of engine args in the platform's canonical spelling. */
export type Config = Record<string, unknown>;

export const CONTRACT_VERSION = "1.7";
