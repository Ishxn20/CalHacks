// Convex components this app installs. The Action Retrier runs Claude's episode labelling with exponential
// backoff and exactly-once completion callbacks, instead of a hand-rolled retry cron.
import actionRetrier from "@convex-dev/action-retrier/convex.config.js";
import { defineApp } from "convex/server";

const app = defineApp();
app.use(actionRetrier);
export default app;
