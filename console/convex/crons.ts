import { cronJobs } from "convex/server";
import { internal } from "./_generated/api";

const crons = cronJobs();
// Retry labelling that failed transiently (rate limits, network) without hammering the API.
crons.interval("retry episode labels", { minutes: 10 }, internal.episodes.retryFailedLabels, {});
crons.interval("prune old arm readings", { hours: 1 }, internal.metrics.prune, {});
export default crons;
