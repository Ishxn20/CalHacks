/**
 * Factory Droid incident worker.
 *
 *   CONVEX_URL=... WORKER_KEY=... FACTORY_API_KEY=... npm run droid
 *
 * Trigger: a live Convex subscription to open incidents. A robot that faults reports an incident;
 * this worker wakes immediately, claims it, and runs Droid on an evidence bundle.
 *
 * What Droid can touch: a throwaway directory holding that incident's telemetry, arm events, the
 * tunable config values and a copy of the arm driver source. Autonomy is off and every permission
 * request is auto-rejected, so it reads but can't edit, run commands or reach the robot. It returns
 * a structured diagnosis and at most one config change; Convex checks the change against the
 * allowlist and bounds, and nothing reaches an arm until a person approves it on the dashboard.
 * Every step it takes is streamed to the incident so the dashboard shows its work.
 */
import { AutonomyLevel, createSession, type DroidSession } from "@factory/droid-sdk/node";
import { ConvexClient } from "convex/browser";
import { mkdtemp, rm, writeFile, copyFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { api } from "../convex/_generated/api.js";
import type { Doc, Id } from "../convex/_generated/dataModel.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(here, "../..");
const CONVEX_URL = process.env.CONVEX_URL ?? process.env.VITE_CONVEX_URL;
const WORKER_KEY = process.env.WORKER_KEY;
const TIMEOUT_MS = 5 * 60_000;

if (!CONVEX_URL || !WORKER_KEY) {
  console.error("set CONVEX_URL (or VITE_CONVEX_URL) and WORKER_KEY");
  process.exit(1);
}

const DIAGNOSIS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["diagnosis", "severity", "safeToResume", "change"],
  properties: {
    diagnosis: { type: "string", description: "What happened and why, citing the telemetry. 2-6 sentences." },
    severity: { type: "string", enum: ["low", "medium", "high"] },
    safeToResume: {
      type: "boolean",
      description: "true only if the arm can be reset and used again without a person inspecting it",
    },
    change: {
      description: "At most one arm.yaml change from tunables.json, or null if a setting isn't the fix",
      anyOf: [
        { type: "null" },
        {
          type: "object",
          additionalProperties: false,
          required: ["key", "from", "to", "rationale"],
          properties: {
            key: { type: "string" },
            from: { type: ["number", "null"] },
            to: { type: "number" },
            rationale: { type: "string" },
          },
        },
      ],
    },
  },
} as const;

type Diagnosis = {
  diagnosis: string;
  severity: "low" | "medium" | "high";
  safeToResume: boolean;
  change: { key: string; from: number | null; to: number; rationale: string } | null;
};

const PROMPT = `You are on call for a fleet of robot arms that hand surgical instruments to surgeons.
One arm just faulted and stopped. This directory is the evidence:

- incident.json: the fault message, the arm's phase, its last job, recent arm events
- telemetry.csv: the last ~60 s of joint positions vs. setpoints (deg), gripper angle and phase, 2 Hz
- tunables.json: every setting you may propose changing, its current value on this arm and its allowed range
- driver.py: the arm controller (the collision guard, gripper ramp and grasp logic live here)
- arm.yaml: the arm's full configuration, for reference

Work out what caused the fault. Read the files; cite times and joint numbers from the telemetry.
Then decide:
- severity: high if a person or the arm could have been hurt or the cause is unknown; low if it's a
  benign, explained trip.
- safeToResume: true only when the cause is understood and a reset is enough.
- change: propose at most ONE setting from tunables.json, inside its range, only if it addresses the
  cause. Otherwise null. A person will review it before it reaches the arm. Never propose disabling
  a safety check.`;

const client = new ConvexClient(CONVEX_URL);
const busy = new Set<string>();

async function step(incidentId: Id<"incidents">, kind: string, text: string) {
  await client.mutation(api.incidents.step, { workerKey: WORKER_KEY!, incidentId, kind, text });
}

async function bundle(inc: Doc<"incidents"> & { robotName: string }, tunables: Record<string, unknown>) {
  const dir = await mkdtemp(path.join(tmpdir(), "incident-"));
  const ctx = JSON.parse(inc.context || "{}");
  const recent: any[] = ctx.recent ?? [];
  const rows = recent.map((s) =>
    [new Date(s.t).toISOString(), s.phase, ...(s.qDeg ?? []), ...(s.qCmdDeg ?? []), s.gripperDeg ?? ""].join(","),
  );
  const header = ["time", "phase", ...[1, 2, 3, 4, 5, 6].map((j) => `q${j}`), ...[1, 2, 3, 4, 5, 6].map((j) => `cmd${j}`), "gripper"];
  await writeFile(path.join(dir, "telemetry.csv"), [header.join(","), ...rows].join("\n"));
  await writeFile(
    path.join(dir, "incident.json"),
    JSON.stringify(
      { robot: inc.robotName, fault: inc.fault, phase: inc.phase, openedAt: new Date(inc.openedAt).toISOString(),
        lastJob: ctx.lastJob ?? null, events: ctx.events ?? [] },
      null,
      2,
    ),
  );
  const current: Record<string, number | null> = ctx.config ?? {};
  await writeFile(
    path.join(dir, "tunables.json"),
    JSON.stringify(
      Object.fromEntries(Object.entries(tunables).map(([k, spec]) => [k, { current: current[k] ?? null, ...(spec as object) }])),
      null,
      2,
    ),
  );
  await copyFile(path.join(REPO, "arm/scrub_arm/driver.py"), path.join(dir, "driver.py"));
  await copyFile(path.join(REPO, "arm/config/arm.yaml"), path.join(dir, "arm.yaml"));
  return dir;
}

/** Droid's subprocess gets no fleet credentials: it can't call Convex even if it tried. */
function droidEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (v !== undefined && !/WORKER_KEY|OPERATOR_KEY|EXPORT_KEY|CONVEX|ROBOT_TOKEN/.test(k)) env[k] = v;
  }
  return env;
}

async function diagnose(inc: Doc<"incidents"> & { robotName: string }) {
  const incidentId = inc._id;
  if (!(await client.mutation(api.incidents.claim, { workerKey: WORKER_KEY!, incidentId }))) return;
  console.log(`[droid] ${inc.robotName}: ${inc.fault}`);
  const tunables = await client.query(api.incidents.tunables, {});
  const dir = await bundle(inc, tunables);
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), TIMEOUT_MS);
  let session: DroidSession | undefined;
  try {
    await step(incidentId, "plan", "Collected telemetry, events, config and driver source; starting read-only diagnosis.");
    session = await createSession({
      cwd: dir,
      env: droidEnv(),
      autonomyLevel: AutonomyLevel.Off,
      autoRejectPermissionRequests: true,
      abortSignal: abort.signal,
    });
    let result: Diagnosis | null = null;
    for await (const msg of session.stream(PROMPT, {
      outputFormat: { type: "json_schema", schema: DIAGNOSIS_SCHEMA as unknown as Record<string, unknown> },
      abortSignal: abort.signal,
    })) {
      switch (msg.type) {
        case "assistant":
          if (msg.text.trim()) await step(incidentId, "thought", msg.text);
          break;
        case "tool_call":
          await step(incidentId, "tool", `${msg.name} ${JSON.stringify(msg.input).slice(0, 300)}`);
          break;
        case "tool_result":
          if (msg.isError) await step(incidentId, "tool_error", `${msg.toolName}: ${String(msg.content).slice(0, 300)}`);
          break;
        case "error":
          await step(incidentId, "error", msg.message);
          break;
        case "result":
          if (!msg.success) throw new Error(msg.error?.message ?? `droid ${msg.subtype}`);
          result = msg.structuredOutput as Diagnosis;
          break;
      }
    }
    if (!result) throw new Error("Droid finished without a diagnosis");
    await client.mutation(api.incidents.propose, {
      workerKey: WORKER_KEY!,
      incidentId,
      diagnosis: result.diagnosis,
      severity: result.severity,
      safeToResume: result.safeToResume,
      change: result.change,
    });
    await step(incidentId, "done", result.change ? `Proposed ${result.change.key} -> ${result.change.to}` : "No config change proposed.");
  } catch (e) {
    const msg = abort.signal.aborted ? `timed out after ${TIMEOUT_MS / 60000} min` : e instanceof Error ? e.message : String(e);
    await client.mutation(api.incidents.fail, { workerKey: WORKER_KEY!, incidentId, error: msg });
  } finally {
    clearTimeout(timer);
    await session?.close().catch(() => {});
    await rm(dir, { recursive: true, force: true });
  }
}

client.onUpdate(api.incidents.open, { workerKey: WORKER_KEY }, (incidents) => {
  for (const inc of incidents) {
    if (busy.has(inc._id)) continue;
    busy.add(inc._id);
    diagnose(inc).finally(() => busy.delete(inc._id));
  }
});
console.log("[droid] watching for incidents");
