import assert from "node:assert/strict";
import test from "node:test";

import { buildBatches } from "./prepare_native_judge_batches.mjs";

test("buildBatches excludes automated and long-output records before batching", () => {
  const finalRecords = [
    { record_id: "r1", item: { domain: "fantasy", item_id: 1 }, text: "one" },
    { record_id: "r2", item: { domain: "fantasy", item_id: 1 }, text: "two" },
    { record_id: "r3", item: { domain: "fantasy", item_id: 1 }, text: "three" },
    { record_id: "r4", item: { domain: "fantasy", item_id: 1 }, text: "four" },
  ];
  const benchmark = [{ domain: "fantasy", item_id: 1, full_instruction: "Write a story." }];
  const constraints = [{ story_id: "fantasy_001", constraints: [{ constraint_id: "fantasy_001_c1" }] }];

  const batches = buildBatches({
    finalRecords,
    automatedScoreIds: new Set(["r1", "r2"]),
    excludedLongIds: new Set(["r2"]),
    benchmark,
    constraints,
    batchSize: 2,
  });

  assert.deepEqual(batches.map((batch) => batch.map((row) => row.record_id)), [["r3", "r4"]]);
  assert.deepEqual(Object.keys(batches[0][0]).sort(), ["constraints", "full_instruction", "record_id", "story_id", "text"]);
  assert.equal(batches[0][0].story_id, "fantasy_001");
});

test("buildBatches rejects a final record that cannot be blinded", () => {
  assert.throws(
    () => buildBatches({
      finalRecords: [{ record_id: "r1", item: { domain: "fantasy", item_id: 1 }, text: "one" }],
      automatedScoreIds: new Set(),
      excludedLongIds: new Set(),
      benchmark: [],
      constraints: [],
      batchSize: 8,
    }),
    /benchmark task/,
  );
});
