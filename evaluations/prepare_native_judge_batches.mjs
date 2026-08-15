import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

function storyId(item) {
  return `${item.domain}_${String(item.item_id).padStart(3, "0")}`;
}

function indexBy(rows, key, description) {
  const indexed = new Map();
  for (const row of rows) {
    if (indexed.has(row[key])) {
      throw new Error(`duplicate ${description}: ${row[key]}`);
    }
    indexed.set(row[key], row);
  }
  return indexed;
}

export function buildBatches({
  finalRecords,
  automatedScoreIds,
  excludedLongIds,
  benchmark,
  constraints,
  batchSize,
}) {
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error("batchSize must be a positive integer");
  }

  const benchmarkByStory = indexBy(
    benchmark.map((task) => ({ ...task, story_id: storyId(task) })),
    "story_id",
    "benchmark story",
  );
  const constraintsByStory = indexBy(constraints, "story_id", "constraint story");
  const finalById = indexBy(finalRecords, "record_id", "final record");
  const blinded = [];

  for (const record of finalById.values()) {
    if (automatedScoreIds.has(record.record_id) || excludedLongIds.has(record.record_id)) {
      continue;
    }
    const id = storyId(record.item);
    const task = benchmarkByStory.get(id);
    const constraintRecord = constraintsByStory.get(id);
    if (!task) {
      throw new Error(`missing benchmark task for ${record.record_id}`);
    }
    if (!constraintRecord) {
      throw new Error(`missing constraints for ${record.record_id}`);
    }
    blinded.push({
      record_id: record.record_id,
      story_id: id,
      full_instruction: task.full_instruction,
      constraints: constraintRecord.constraints,
      text: record.text,
    });
  }

  return Array.from({ length: Math.ceil(blinded.length / batchSize) }, (_, index) =>
    blinded.slice(index * batchSize, (index + 1) * batchSize),
  );
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, "utf8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function requiredOption(args, name) {
  const index = args.indexOf(name);
  if (index < 0 || !args[index + 1]) {
    throw new Error(`missing required option ${name}`);
  }
  return args[index + 1];
}

function main(args) {
  const outputDir = requiredOption(args, "--output");
  const batchSize = Number(requiredOption(args, "--batch-size"));
  if (fs.existsSync(outputDir)) {
    throw new Error(`refusing to overwrite existing output directory: ${outputDir}`);
  }

  const finalRecords = readJsonl("runs/results/all_results.jsonl").filter((record) => record.is_final);
  const automatedScoreIds = new Set(readJsonl("evaluations/scores_auto.jsonl").map((row) => row.record_id));
  const excludedLongIds = new Set(readJsonl("evaluations/excluded_for_manual_annotation.jsonl").map((row) => row.record_id));
  const benchmark = JSON.parse(fs.readFileSync("runs/benchmark_data.json", "utf8"));
  const constraints = readJsonl("evaluations/constraints.jsonl");
  const batches = buildBatches({
    finalRecords,
    automatedScoreIds,
    excludedLongIds,
    benchmark,
    constraints,
    batchSize,
  });

  fs.mkdirSync(outputDir);
  const manifest = [];
  for (const [index, batch] of batches.entries()) {
    const filename = `batch-${String(index + 1).padStart(3, "0")}.json`;
    fs.writeFileSync(path.join(outputDir, filename), `${JSON.stringify(batch)}\n`, "utf8");
    manifest.push({ batch: index + 1, file: filename, record_ids: batch.map((row) => row.record_id) });
  }
  fs.writeFileSync(path.join(outputDir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify({ batches: batches.length, records: manifest.flatMap((row) => row.record_ids).length })}\n`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2));
}
