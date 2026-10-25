import { ActionRetrier } from "@convex-dev/action-retrier";
import { components } from "./_generated/api";

// Labelling calls Claude with several images: back off from 15 s (rate limits, 5xx) and give up after 4 tries.
export const retrier = new ActionRetrier(components.actionRetrier, {
  initialBackoffMs: 15_000,
  base: 2,
  maxFailures: 4,
});
