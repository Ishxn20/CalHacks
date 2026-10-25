import { ConvexError } from "convex/values";
import type { Id } from "./_generated/dataModel";
import type { MutationCtx } from "./_generated/server";

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time-ish comparison so a wrong key doesn't leak how many characters matched. */
function sameSecret(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * Anything that moves an arm or changes its configuration needs the operator key
 * (Convex env var OPERATOR_KEY). Fails closed when the key isn't configured.
 */
export function requireOperator(key: string): string {
  const expected = process.env.OPERATOR_KEY;
  if (!expected) throw new ConvexError("OPERATOR_KEY is not set on this deployment; operator actions are disabled");
  if (!sameSecret(key, expected)) throw new ConvexError("wrong operator key");
  return "operator";
}

/** The Droid incident worker authenticates with WORKER_KEY; it can write diagnoses, never commands. */
export function requireWorker(key: string): string {
  const expected = process.env.WORKER_KEY;
  if (!expected) throw new ConvexError("WORKER_KEY is not set on this deployment");
  if (!sameSecret(key, expected)) throw new ConvexError("wrong worker key");
  return "droid-worker";
}

export async function audit(
  ctx: MutationCtx,
  actor: string,
  action: string,
  robotId: Id<"robots"> | null,
  detail: unknown,
): Promise<void> {
  await ctx.db.insert("audit", {
    t: Date.now(),
    actor,
    action,
    robotId,
    detail: typeof detail === "string" ? detail : JSON.stringify(detail),
  });
}

/**
 * The only arm.yaml settings a remote fix may touch, with hard bounds. Everything else
 * (joint limits, gains, motor IDs, poses) needs a person at the arm.
 */
export const TUNABLE: Record<string, { min: number; max: number; unit: string; what: string }> = {
  "motion.speed_scale": { min: 0.1, max: 0.6, unit: "", what: "global speed multiplier" },
  "motion.approach_speed_scale": { min: 0.05, max: 0.4, unit: "", what: "final approach speed multiplier" },
  "motion.tracking_error_limit_deg": { min: 10, max: 30, unit: "deg", what: "collision guard threshold" },
  "motion.settle_timeout_s": { min: 0.5, max: 4, unit: "s", what: "time allowed to settle on a target" },
  "gripper.move_torque_nm": { min: 0.5, max: 2.0, unit: "Nm", what: "jaw torque cap while moving" },
  "gripper.grip_torque_nm": { min: 0.3, max: 1.5, unit: "Nm", what: "squeeze torque when grasping" },
  "gripper.holding_min_gap_deg": { min: 1.0, max: 6.0, unit: "deg", what: "stall gap that counts as holding" },
  "pick.contact_deviation_deg": { min: 0.8, max: 4.0, unit: "deg", what: "joint lag that counts as touching" },
};

export function checkChange(key: string, to: number): void {
  const spec = TUNABLE[key];
  if (!spec) throw new ConvexError(`${key} is not remotely tunable`);
  if (!Number.isFinite(to) || to < spec.min || to > spec.max) {
    throw new ConvexError(`${key} must be within ${spec.min}..${spec.max} ${spec.unit}`.trim());
  }
}
