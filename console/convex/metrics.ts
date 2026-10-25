import { internalMutation, query } from "./_generated/server";
import { internal } from "./_generated/api";
import { v, type Infer } from "convex/values";
import { point, sample } from "./schema";

// shortcut: fixed 1-day retention, no longer-term rollups; add hourly rollups if older history is needed.
const RETENTION_S = 24 * 3600;
const PRUNE_BATCH = 200; // raw batches are ~15 KB, so this stays far below a mutation's read limit
const MAX_RAW_S = 600;

type Sample = Infer<typeof sample>;
type Point = Infer<typeof point>;

const maxOf = (a: number | null, b: number | null) => (a === null ? b : b === null ? a : Math.max(a, b));
const r2 = (x: number) => Math.round(x * 100) / 100;

// One point per whole second. Lag (|q - q_cmd|) and |torque| keep the worst value in the second, so a
// contact spike shows up on a chart even though most of its 20 Hz samples are not drawn.
export function rollUp(samples: Sample[]): Point[] {
  const out: Point[] = [];
  for (const s of samples) {
    const t = Math.floor(s.t);
    let p = out[out.length - 1];
    if (!p || p.t !== t) {
      p = { t, q: s.q, lag: s.q.map(() => null), tau: s.tau.map(() => null), gripper: s.gripper, moving: false, phase: s.phase, fault: null };
      out.push(p);
    }
    const q = s.q, qc = s.q_cmd, tau = s.tau;
    p.lag = p.lag.map((m, i) => maxOf(m, q[i] == null || qc[i] == null ? null : r2(Math.abs(q[i]! - qc[i]!))));
    p.tau = p.tau.map((m, i) => maxOf(m, tau[i] == null ? null : Math.abs(tau[i]!)));
    p.q = q;
    p.gripper = s.gripper;
    p.phase = s.phase;
    p.moving ||= s.moving;
    p.fault = s.fault ?? p.fault;
  }
  return out;
}

// Internal: only the console (deploy key) writes readings, the same rule as events.add.
export const record = internalMutation({
  args: { t0: v.number(), t1: v.number(), source: v.string(), samples: v.array(sample) },
  handler: async (ctx, args) => {
    await ctx.db.insert("metrics", args);
    await ctx.db.insert("metricPoints", { t0: args.t0, source: args.source, points: rollUp(args.samples) });
  },
});

export const prune = internalMutation({
  args: {},
  handler: async (ctx) => {
    const cutoff = Date.now() / 1000 - RETENTION_S;
    let more = false;
    for (const table of ["metrics", "metricPoints"] as const) {
      const old = await ctx.db.query(table).withIndex("by_t0", (q) => q.lt("t0", cutoff)).take(PRUNE_BATCH);
      for (const d of old) await ctx.db.delete(d._id);
      more ||= old.length === PRUNE_BATCH;
    }
    if (more) await ctx.scheduler.runAfter(0, internal.metrics.prune, {});
  },
});

// Chart history from `from` (unix s). The dashboard loads this once, then follows `latest`, so a new batch
// re-reads one small document instead of the whole window.
export const points = query({
  args: { from: v.number() },
  handler: async (ctx, { from }) => {
    const rows = await ctx.db.query("metricPoints").withIndex("by_t0", (q) => q.gte("t0", from - 10)).take(800);
    return rows.flatMap((r) => r.points.filter((p) => p.t >= from));
  },
});

export const latest = query({
  args: {},
  handler: async (ctx) => (await ctx.db.query("metricPoints").withIndex("by_t0").order("desc").first())?.points ?? [],
});

// Every 20 Hz reading in [from, to], at most 10 minutes of them (the dashboard's CSV download).
export const raw = query({
  args: { from: v.number(), to: v.number() },
  handler: async (ctx, { from, to }) => {
    const end = Math.min(to, from + MAX_RAW_S);
    const rows = await ctx.db
      .query("metrics")
      .withIndex("by_t0", (q) => q.gte("t0", from - 10).lte("t0", end))
      .collect();
    return rows.flatMap((r) => r.samples.filter((s) => s.t >= from && s.t <= end));
  },
});
