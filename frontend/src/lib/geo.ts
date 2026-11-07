/**
 * Geo validation utilities — shared across all map components.
 * Filters out NaN / Infinity coordinates before they reach Leaflet.
 */

/** Type guard: is `p` a valid [lat, lng] tuple with finite numbers? */
export const validLL = (p: unknown): p is [number, number] =>
  Array.isArray(p) &&
  p.length >= 2 &&
  Number.isFinite((p as number[])[0]) &&
  Number.isFinite((p as number[])[1]);

/** Fallback centre — Depot Quận 1, TP.HCM */
export const FALLBACK_CENTER: [number, number] = [10.7769, 106.7009];

/**
 * Sanitise an array of coordinate tuples:
 * keeps only entries where both lat and lng are finite numbers.
 */
export function filterValidCoords(pts: [number, number][]): [number, number][] {
  return pts.filter(validLL);
}

