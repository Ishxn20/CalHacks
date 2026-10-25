import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();
// The live event stream is operational history; keep a week. (Episodes and audit anchors are kept.)
crons.interval("prune old events", { hours: 1 }, internal.events.prune, {});
crons.interval("prune old arm readings", { hours: 1 }, internal.metrics.prune, {});
export default crons;
