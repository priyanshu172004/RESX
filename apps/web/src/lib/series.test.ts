/**
 * The colour rules, asserted.
 *
 * These are the failures that never throw. A cycled hue silently tells a reader
 * that series nine and series one are the same thing; a status step reused for
 * "series 4" teaches them that red means refuted and then shows red meaning
 * West Region. Both render perfectly and both are wrong, so nothing but a test
 * catches them.
 */

import { describe, expect, it } from "vitest";

import {
  ALL_PAIRS_SERIES_CAP,
  SERIES_SLOT_COUNT,
  assignChartSlots,
  capForAllPairs,
  needsLightModeRelief,
  seriesColor,
  seriesSlot,
  statusColor,
} from "./series";

describe("slot assignment", () => {
  it("gives one key the same slot every time", () => {
    // Colour follows the entity, not its position. A filter that changes the
    // series count must not repaint the survivors.
    const first = seriesSlot("revenue");
    expect(seriesSlot("revenue")).toBe(first);
    expect(seriesSlot("revenue")).toBe(first);
  });

  it("keeps every slot inside the palette", () => {
    const keys = Array.from({ length: 50 }, (_, i) => `series_${i}`);
    for (const key of keys) {
      const slot = seriesSlot(key);
      expect(slot).toBeGreaterThanOrEqual(1);
      expect(slot).toBeLessThanOrEqual(SERIES_SLOT_COUNT);
    }
  });

  it("never invents a ninth colour", () => {
    // The palette is validated at eight steps. A generated hue has passed no
    // contrast or CVD check, and it would look plausible while failing both.
    const colours = new Set(
      Array.from({ length: 40 }, (_, i) => seriesColor(`k${i}`)),
    );
    expect(colours.size).toBeLessThanOrEqual(SERIES_SLOT_COUNT);
  });

  it("returns a token, never a literal hex", () => {
    // Themes swap the steps underneath. A baked hex would stay light-mode blue
    // on a dark surface.
    expect(seriesColor("revenue")).toMatch(/^var\(--chart-[1-8]\)$/);
  });
});

describe("status palette", () => {
  it("is reserved and separate from the series palette", () => {
    const status = (["good", "warning", "serious", "critical"] as const).map(
      statusColor,
    );
    const series = Array.from({ length: 40 }, (_, i) => seriesColor(`k${i}`));
    for (const colour of status) {
      expect(series).not.toContain(colour);
    }
  });

  it("names each role distinctly", () => {
    const seen = new Set(
      (["good", "warning", "serious", "critical"] as const).map(statusColor),
    );
    expect(seen.size).toBe(4);
  });
});

describe("capForAllPairs", () => {
  it("returns what was folded rather than dropping it", () => {
    // The tail is part of the answer. Silently discarding it answers "how
    // concentrated is this" wrongly.
    const series = Array.from({ length: 7 }, (_, i) => ({ key: `s${i}` }));
    const { rendered, folded } = capForAllPairs(series);
    expect(rendered).toHaveLength(ALL_PAIRS_SERIES_CAP);
    expect(rendered.length + folded.length).toBe(series.length);
  });

  it("leaves a small set alone", () => {
    const series = [{ key: "a" }, { key: "b" }];
    const { rendered, folded } = capForAllPairs(series);
    expect(rendered).toHaveLength(2);
    expect(folded).toHaveLength(0);
  });
});

describe("assignChartSlots", () => {
  it("maps every key it is given", () => {
    const keys = ["revenue", "margin", "units"];
    const assigned = assignChartSlots(keys);
    for (const key of keys) {
      expect(assigned[key]).toBeDefined();
    }
  });

  it("assigns consecutive slots from 1", () => {
    // Load-bearing, and the reason this function exists rather than calling
    // `seriesColor` per key. The palette is validated on its *adjacent* pair
    // list, so slot 3 beside slot 6 is a pair nothing ever checked — and it
    // reads as two greens. Consecutive-from-1 is the only assignment the
    // validation actually covers.
    const assigned = assignChartSlots(["alpha", "beta", "gamma", "delta"]);
    const slots = Object.values(assigned)
      .map((token) => Number(/--chart-(\d)/.exec(token)?.[1]))
      .sort((a, b) => a - b);
    expect(slots).toEqual([1, 2, 3, 4]);
  });

  it("gives a registered key its registered colour when the slot is free", () => {
    // So a column actually called "revenue" gets the revenue colour rather
    // than whatever position it happens to occupy.
    const assigned = assignChartSlots(["revenue", "other_a", "other_b"]);
    expect(assigned["revenue"]).toBe(seriesColor("revenue"));
  });

  it("never assigns two series the same colour", () => {
    // Two series sharing a hue is indistinguishable from one series, and the
    // legend would state something the chart contradicts.
    const keys = ["a", "b", "c", "d", "e", "f"];
    const assigned = assignChartSlots(keys);
    expect(new Set(Object.values(assigned)).size).toBe(keys.length);
  });
});

describe("needsLightModeRelief", () => {
  it("answers for any set without throwing", () => {
    // A contrast WARN obliges labels or a table view; it is not dismissable,
    // so the caller must always get a definite answer.
    expect(typeof needsLightModeRelief(["revenue"])).toBe("boolean");
    expect(typeof needsLightModeRelief([])).toBe("boolean");
  });
});
