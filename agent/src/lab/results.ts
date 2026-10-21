// What a finished benchmark becomes in the ledger: the platform's objective
// and feasibility, plus every module's results in measures.

import type { BenchmarkOut } from "../contract/types.js";
import { scenariosOf } from "./measures.js";
import { type Measurement, now } from "./state.js";

export function measurementOf(bench: BenchmarkOut): Measurement {
  const summary = bench.summary;
  return {
    benchId: bench.id,
    objective: summary?.objective_value ?? null,
    feasible: Boolean(summary?.feasible),
    breaches: summary?.breaches ?? [],
    scenarios: scenariosOf(bench),
    at: now(),
  };
}
