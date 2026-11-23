import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { runInputs, type InputEvent } from "./engine";
import type { Chart, JudgeEvent, Stats } from "./types";

interface Fixture {
  name: string;
  chart: Chart;
  inputs: InputEvent[];
  expected: { events: JudgeEvent[]; stats: Stats };
}

const dir = join(__dirname, "../../../tests/fixtures/judge");
const fixtures: Fixture[] = readdirSync(dir)
  .filter((f) => f.endsWith(".json"))
  .map((f) => JSON.parse(readFileSync(join(dir, f), "utf-8")) as Fixture);

describe("judge engine matches Python golden fixtures", () => {
  it("has fixtures", () => expect(fixtures.length).toBeGreaterThan(3));
  for (const fx of fixtures) {
    it(fx.name, () => {
      const { events, stats } = runInputs(fx.chart, fx.inputs);
      expect(events).toEqual(fx.expected.events);
      expect(stats).toEqual(fx.expected.stats);
    });
  }
});
