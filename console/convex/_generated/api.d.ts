/* eslint-disable */
/**
 * Generated `api` utility.
 *
 * THIS CODE IS AUTOMATICALLY GENERATED.
 *
 * To regenerate, run `npx convex dev`.
 * @module
 */

import type * as audit from "../audit.js";
import type * as crons from "../crons.js";
import type * as episodes from "../episodes.js";
import type * as events from "../events.js";
import type * as http from "../http.js";
import type * as instruments from "../instruments.js";
import type * as labeling from "../labeling.js";
import type * as metrics from "../metrics.js";
import type * as proposals from "../proposals.js";
import type * as retrier from "../retrier.js";

import type {
  ApiFromModules,
  FilterApi,
  FunctionReference,
} from "convex/server";

declare const fullApi: ApiFromModules<{
  audit: typeof audit;
  crons: typeof crons;
  episodes: typeof episodes;
  events: typeof events;
  http: typeof http;
  instruments: typeof instruments;
  labeling: typeof labeling;
  metrics: typeof metrics;
  proposals: typeof proposals;
  retrier: typeof retrier;
}>;

/**
 * A utility for referencing Convex functions in your app's public API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = api.myModule.myFunction;
 * ```
 */
export declare const api: FilterApi<
  typeof fullApi,
  FunctionReference<any, "public">
>;

/**
 * A utility for referencing Convex functions in your app's internal API.
 *
 * Usage:
 * ```js
 * const myFunctionReference = internal.myModule.myFunction;
 * ```
 */
export declare const internal: FilterApi<
  typeof fullApi,
  FunctionReference<any, "internal">
>;

export declare const components: {
  actionRetrier: import("@convex-dev/action-retrier/_generated/component.js").ComponentApi<"actionRetrier">;
};
