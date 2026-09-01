/**
 * Reproducible matched evaluator-validation workflow for the existing blind
 * 30-pair sample. It writes only sanitized A/B case material, frozen model
 * judgments, ordinal analysis artifacts, and reports. It never reads human
 * annotations unless the explicit `analyze-human` command is invoked after
 * both model passes have been frozen.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const OUTPUT_DIR = path.join(ROOT, "evaluations", "human_validation");
const MANIFEST_PATH = path.join(ROOT, "evaluations", ".tmp_human_review", "blind_human_review_manifest.json");
const CASE_PACK_PATH = path.join(ROOT, "evaluations", ".tmp_human_review", "blind_case_pack.js");
const PASSES = new Set(["primary", "reversed"]);
const SCALE = ["A clearly better", "A slightly better", "Tie", "B slightly better", "B clearly better"];
const SCALE_SET = new Set(SCALE);
const ORDINAL = new Map(SCALE.map((label, index) => [label, index - 2]));
const HUMAN_EXPORT_LABELS = new Map([
  ["a_clear", "A clearly better"],
  ["a_slight", "A slightly better"],
  ["tie", "Tie"],
  ["b_slight", "B slightly better"],
  ["b_clear", "B clearly better"],
]);
const CASE_FIELDS = new Set(["case_id", "instruction", "constraints", "response_A", "response_B"]);
const CONSTRAINT_FIELDS = new Set(["constraint_id", "requirement"]);
const JUDGMENT_FIELDS = new Set([
  "case_id", "constraint_following", "constraint_reason", "creative_quality", "creative_reason",
]);
const FORBIDDEN_METADATA_FIELDS = new Set([
  "model", "model_id", "model_name", "condition", "full", "sharded", "human", "human_notes",
  "human_ratings", "completion_metadata", "automated_score", "automated_scores", "judge_score",
  "judge_scores", "sampling_stratum", "seed", "quantization", "quant_file", "context_length",
  "provenance", "record_id", "full_record_id", "sharded_record_id", "story_id",
]);

function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function atomicWrite(filePath, text) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = path.join(path.dirname(filePath), `.${path.basename(filePath)}.${process.pid}.${Date.now()}.tmp`);
  fs.writeFileSync(temporary, text, "utf8");
  fs.renameSync(temporary, filePath);
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, "utf8").split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

function assertString(value, label) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string`);
}

function assertExactFields(value, fields, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  for (const key of Object.keys(value)) if (!fields.has(key)) throw new Error(`${label} contains unexpected field: ${key}`);
  for (const key of fields) if (!(key in value)) throw new Error(`${label} is missing required field: ${key}`);
}

function scanMetadataKeys(value, location = "root") {
  if (Array.isArray(value)) return value.forEach((item, index) => scanMetadataKeys(item, `${location}[${index}]`));
  if (!value || typeof value !== "object") return;
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_METADATA_FIELDS.has(key.toLowerCase())) {
      throw new Error(`sanitized input contains forbidden metadata field at ${location}.${key}`);
    }
    scanMetadataKeys(child, `${location}.${key}`);
  }
}

function parseCasePack(source) {
  const prefix = "window.BLIND_CASES=";
  if (!source.startsWith(prefix)) throw new Error("blind case pack does not use its expected literal format");
  return JSON.parse(source.slice(prefix.length).trim().replace(/;$/, ""));
}

function validateCases(cases) {
  if (!Array.isArray(cases) || cases.length !== 30) throw new Error("sanitized input must contain exactly 30 cases");
  const ids = new Set();
  for (const [index, row] of cases.entries()) {
    assertExactFields(row, CASE_FIELDS, `case ${index + 1}`);
    assertString(row.case_id, `case ${index + 1} case_id`);
    if (ids.has(row.case_id)) throw new Error(`duplicate case_id: ${row.case_id}`);
    ids.add(row.case_id);
    assertString(row.instruction, `${row.case_id} instruction`);
    assertString(row.response_A, `${row.case_id} response_A`);
    assertString(row.response_B, `${row.case_id} response_B`);
    if (!Array.isArray(row.constraints) || !row.constraints.length) throw new Error(`${row.case_id} has no constraints`);
    row.constraints.forEach((constraint, constraintIndex) => {
      assertExactFields(constraint, CONSTRAINT_FIELDS, `${row.case_id} constraint ${constraintIndex + 1}`);
      assertString(constraint.constraint_id, `${row.case_id} constraint_id`);
      assertString(constraint.requirement, `${row.case_id} requirement`);
    });
  }
  scanMetadataKeys(cases);
  return { cases: cases.length, responses: cases.length * 2 };
}

function buildPrimaryCases() {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
  const pack = parseCasePack(fs.readFileSync(CASE_PACK_PATH, "utf8"));
  if (!Array.isArray(manifest) || manifest.length !== 30 || !Array.isArray(pack) || pack.length !== 30) {
    throw new Error("blind source must contain exactly 30 manifest and case-pack rows");
  }
  const manifestIds = new Set(manifest.map((row) => row?.case_id));
  const packIds = new Set(pack.map((row) => row?.case_id));
  if (manifestIds.size !== 30 || packIds.size !== 30 || [...manifestIds].some((id) => !packIds.has(id))) {
    throw new Error("blind manifest and case pack do not identify the same 30 cases");
  }
  const cases = pack.map((row) => ({
    case_id: row.case_id,
    instruction: row.full_instruction,
    constraints: row.constraints.map(({ constraint_id, requirement }) => ({ constraint_id, requirement })),
    response_A: row.response_a,
    response_B: row.response_b,
  }));
  validateCases(cases);
  return cases;
}

function reverseCases(cases) {
  const reversed = cases.map((row) => ({ ...row, response_A: row.response_B, response_B: row.response_A }));
  validateCases(reversed);
  return reversed;
}

function casePath(pass) {
  return path.join(OUTPUT_DIR, `sanitized_${pass}_cases.jsonl`);
}

function judgmentPath(pass) {
  return path.join(OUTPUT_DIR, `model_pairwise_${pass}.jsonl`);
}

function validateJudgments(cases, judgments) {
  validateCases(cases);
  if (!Array.isArray(judgments) || judgments.length !== 30) throw new Error("judgments must contain exactly 30 records");
  const expected = new Set(cases.map((row) => row.case_id));
  const byCase = new Map();
  for (const row of judgments) {
    assertExactFields(row, JUDGMENT_FIELDS, "judgment");
    assertString(row.case_id, "judgment case_id");
    if (!expected.has(row.case_id)) throw new Error(`unexpected judgment case_id: ${row.case_id}`);
    if (byCase.has(row.case_id)) throw new Error(`duplicate judgment case_id: ${row.case_id}`);
    if (!SCALE_SET.has(row.constraint_following)) throw new Error(`invalid constraint label for ${row.case_id}`);
    if (!SCALE_SET.has(row.creative_quality)) throw new Error(`invalid creative label for ${row.case_id}`);
    assertString(row.constraint_reason, `${row.case_id} constraint_reason`);
    assertString(row.creative_reason, `${row.case_id} creative_reason`);
    byCase.set(row.case_id, row);
  }
  return cases.map((row) => byCase.get(row.case_id));
}

function prepare() {
  const primary = buildPrimaryCases();
  const reversed = reverseCases(primary);
  const validation = {};
  for (const [pass, cases] of [["primary", primary], ["reversed", reversed]]) {
    const serialized = `${cases.map((row) => JSON.stringify(row)).join("\n")}\n`;
    atomicWrite(casePath(pass), serialized);
    validation[pass] = { ...validateCases(cases), sha256: sha256(serialized) };
  }
  atomicWrite(path.join(OUTPUT_DIR, "sanitization_validation.json"), `${JSON.stringify({
    cases_per_pass: 30,
    responses_per_pass: 60,
    forbidden_metadata_fields: [...FORBIDDEN_METADATA_FIELDS].sort(),
    contains_forbidden_metadata_fields: false,
    passes: validation,
  }, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(validation)}\n`);
}

function freeze(pass, candidatePath) {
  if (!PASSES.has(pass)) throw new Error("pass must be primary or reversed");
  const cases = readJsonl(casePath(pass));
  const judgments = validateJudgments(cases, readJsonl(candidatePath));
  const serialized = `${judgments.map((row) => JSON.stringify(row)).join("\n")}\n`;
  atomicWrite(judgmentPath(pass), serialized);
  atomicWrite(path.join(OUTPUT_DIR, `model_pairwise_${pass}_freeze.json`), `${JSON.stringify({
    pass,
    cases: judgments.length,
    responses: judgments.length * 2,
    cases_sha256: sha256(fs.readFileSync(casePath(pass), "utf8")),
    judgments_sha256: sha256(serialized),
    comparison_with_humans_performed: false,
  }, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ pass, cases: 30, responses: 60, sha256: sha256(serialized) })}\n`);
}

function invertLabel(label) {
  const value = ORDINAL.get(label);
  if (value === undefined) throw new Error(`cannot invert unknown label: ${label}`);
  return SCALE[value * -1 + 2];
}

function direction(value) {
  return Math.sign(value);
}

function positionConsistency() {
  const primary = readJsonl(judgmentPath("primary"));
  const reversed = readJsonl(judgmentPath("reversed"));
  const reversedByCase = new Map(reversed.map((row) => [row.case_id, row]));
  const dimensions = [
    ["constraint_following", "constraint_following"],
    ["creative_quality", "creative_quality"],
  ];
  const summary = {};
  const changes = [];
  for (const [dimension, field] of dimensions) {
    let directionalMatches = 0;
    let exactMatches = 0;
    for (const row of primary) {
      const swapped = reversedByCase.get(row.case_id);
      if (!swapped) throw new Error(`missing reversed judgment for ${row.case_id}`);
      const translated = invertLabel(swapped[field]);
      const originalValue = ORDINAL.get(row[field]);
      const translatedValue = ORDINAL.get(translated);
      if (direction(originalValue) === direction(translatedValue)) directionalMatches += 1;
      if (row[field] === translated) exactMatches += 1;
      if (direction(originalValue) !== direction(translatedValue)) {
        changes.push({ case_id: row.case_id, dimension, primary: row[field], reversed_translated_to_primary: translated });
      }
    }
    summary[dimension] = {
      n: primary.length,
      directional_consistency: directionalMatches / primary.length,
      exact_five_level_consistency: exactMatches / primary.length,
      directional_changes: primary.length - directionalMatches,
    };
  }
  const allDirectional = Object.values(summary).reduce((total, item) => total + item.directional_consistency * item.n, 0);
  const allExact = Object.values(summary).reduce((total, item) => total + item.exact_five_level_consistency * item.n, 0);
  const artifact = {
    orientation: "Reversed labels are transformed back to the original A/B orientation before comparison.",
    per_dimension: summary,
    combined: { n: primary.length * 2, directional_consistency: allDirectional / (primary.length * 2), exact_five_level_consistency: allExact / (primary.length * 2) },
    directional_preference_changes: changes,
  };
  atomicWrite(path.join(OUTPUT_DIR, "position_consistency_summary.json"), `${JSON.stringify(artifact, null, 2)}\n`);
  return artifact;
}

function ordinalRows() {
  const primary = readJsonl(judgmentPath("primary"));
  const reversed = readJsonl(judgmentPath("reversed"));
  const reversedByCase = new Map(reversed.map((row) => [row.case_id, row]));
  const rows = primary.map((row) => {
    const swapped = reversedByCase.get(row.case_id);
    return {
      case_id: row.case_id,
      primary_constraint_following: row.constraint_following,
      primary_constraint_ordinal: ORDINAL.get(row.constraint_following),
      primary_creative_quality: row.creative_quality,
      primary_creative_ordinal: ORDINAL.get(row.creative_quality),
      reversed_constraint_following_in_original_orientation: invertLabel(swapped.constraint_following),
      reversed_constraint_ordinal_in_original_orientation: -ORDINAL.get(swapped.constraint_following),
      reversed_creative_quality_in_original_orientation: invertLabel(swapped.creative_quality),
      reversed_creative_ordinal_in_original_orientation: -ORDINAL.get(swapped.creative_quality),
    };
  });
  atomicWrite(path.join(OUTPUT_DIR, "ordinal_model_judgments.jsonl"), `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
}

function report() {
  const position = positionConsistency();
  ordinalRows();
  const humanSummaryPath = path.join(OUTPUT_DIR, "human_model_agreement_summary.json");
  const humanSummary = fs.existsSync(humanSummaryPath)
    ? JSON.parse(fs.readFileSync(humanSummaryPath, "utf8"))
    : null;
  const percentage = (value) => `${(value * 100).toFixed(1)}%`;
  const humanLines = humanSummary ? [
    `Completed human cases: ${humanSummary.human_cases_completed}. Primary-order model judgments are compared with the same A/B assignment.`,
    `- Constraint following: exact ${percentage(humanSummary.constraint_following.exact_five_level_agreement)}; directional ${percentage(humanSummary.constraint_following.directional_agreement)}; quadratic weighted kappa ${humanSummary.constraint_following.weighted_cohens_kappa_quadratic?.toFixed(3) ?? "undefined"}; collapsed kappa ${humanSummary.constraint_following.collapsed_cohens_kappa?.toFixed(3) ?? "undefined"}; mean absolute disagreement ${humanSummary.constraint_following.mean_absolute_disagreement.toFixed(3)}; severe disagreement ${percentage(humanSummary.constraint_following.severe_disagreement_rate)}.`,
    `- Creative-writing quality: exact ${percentage(humanSummary.creative_writing_quality.exact_five_level_agreement)}; directional ${percentage(humanSummary.creative_writing_quality.directional_agreement)}; quadratic weighted kappa ${humanSummary.creative_writing_quality.weighted_cohens_kappa_quadratic?.toFixed(3) ?? "undefined"}; collapsed kappa ${humanSummary.creative_writing_quality.collapsed_cohens_kappa?.toFixed(3) ?? "undefined"}; mean absolute disagreement ${humanSummary.creative_writing_quality.mean_absolute_disagreement.toFixed(3)}; severe disagreement ${percentage(humanSummary.creative_writing_quality.severe_disagreement_rate)}.`,
    "See `human_model_agreement_summary.json` and `human_model_disagreements.md` for the completed-case data and diagnostic table.",
  ] : [
    "No human annotation export is present in this repository, so exact agreement, directional agreement, weighted Cohen's kappa, collapsed Cohen's kappa, mean absolute disagreement, severe-disagreement rate, and the per-case disagreement table have not been computed. This is a data-availability limitation, not an exclusion of cases.",
  ];
  const lines = [
    "# Matched Human–Model Pairwise Validation",
    "",
    "## Raw observations",
    "",
    "- The fixed blind sample contains 30 matched response pairs (60 responses).",
    "- The primary pass preserves the app's existing Response A/Response B assignment.",
    "- The reversed pass swaps the same two responses within every case; instruction and atomic constraints are unchanged.",
    "- Both model passes were validated and frozen before any human annotations were loaded.",
    "",
    "## Computed statistics",
    "",
    "### Position consistency",
    "",
    `- Constraint following: directional ${(position.per_dimension.constraint_following.directional_consistency * 100).toFixed(1)}%; exact five-level ${(position.per_dimension.constraint_following.exact_five_level_consistency * 100).toFixed(1)}% (N=30).`,
    `- Creative-writing quality: directional ${(position.per_dimension.creative_quality.directional_consistency * 100).toFixed(1)}%; exact five-level ${(position.per_dimension.creative_quality.exact_five_level_consistency * 100).toFixed(1)}% (N=30).`,
    `- Combined across both decisions: directional ${(position.combined.directional_consistency * 100).toFixed(1)}%; exact five-level ${(position.combined.exact_five_level_consistency * 100).toFixed(1)}% (N=60).`,
    `- Directional preference changes after swapping: ${position.directional_preference_changes.length}. See \`position_consistency_summary.json\` for the complete list.`,
    "",
    "### Human–model agreement",
    "",
    ...humanLines,
    "",
    "## Interpretation",
    "",
    "Position consistency describes whether the model's comparative direction is stable under a pure A/B label swap. It does not establish agreement with human reviewers and must not be interpreted as validity evidence by itself.",
    "",
    "## Limitations",
    "",
    "- The sample is N=30 and is a matched evaluator-validation study, not a rerun of the 1,920-output pointwise evaluation.",
    humanSummary
      ? `- The supplied human export contains ${humanSummary.human_cases_completed} completed cases; all human-model statistics are based on N=${humanSummary.human_cases_completed}, not the full 30-case sample.`
      : "- The repository does not contain the human export, so no human-model estimate can yet be reported.",
    "- The repository's hidden manifest records selection strata, but it does not identify a recoverable development/pilot versus held-out boundary. This report therefore does not claim that all 30 cases were untouched held-out validation data.",
    "- Categorical labels remain authoritative; the -2 to +2 representation in `ordinal_model_judgments.jsonl` is for analysis only.",
    "",
  ];
  atomicWrite(path.join(OUTPUT_DIR, "methodology_results.md"), `${lines.join("\n")}\n`);
}

function directionLabel(label) {
  const value = ORDINAL.get(label);
  if (value === undefined) throw new Error(`unknown ordinal label: ${label}`);
  return value < 0 ? "A" : value > 0 ? "B" : "Tie";
}

function normalizeHumanLabel(label, caseId, field) {
  if (SCALE_SET.has(label)) return label;
  const normalized = HUMAN_EXPORT_LABELS.get(label);
  if (!normalized) throw new Error(`human export has invalid ${field} label for ${caseId}`);
  return normalized;
}

function kappa(observedPairs, categories, weight) {
  if (!observedPairs.length) return null;
  const index = new Map(categories.map((category, position) => [category, position]));
  const matrix = Array.from({ length: categories.length }, () => Array(categories.length).fill(0));
  for (const [human, model] of observedPairs) matrix[index.get(human)][index.get(model)] += 1;
  const n = observedPairs.length;
  const rowTotals = matrix.map((row) => row.reduce((total, value) => total + value, 0));
  const columnTotals = categories.map((_, column) => matrix.reduce((total, row) => total + row[column], 0));
  let observed = 0;
  let expected = 0;
  for (let row = 0; row < categories.length; row += 1) {
    for (let column = 0; column < categories.length; column += 1) {
      const currentWeight = weight(row, column, categories.length);
      observed += currentWeight * matrix[row][column] / n;
      expected += currentWeight * rowTotals[row] * columnTotals[column] / (n * n);
    }
  }
  return Math.abs(1 - expected) < Number.EPSILON ? null : (observed - expected) / (1 - expected);
}

function statistics(humanRows, modelByCase, humanField, modelField) {
  const pairs = humanRows.map((human) => [human[humanField], modelByCase.get(human.case_id)[modelField]]);
  const exact = pairs.filter(([human, model]) => human === model).length;
  const directional = pairs.filter(([human, model]) => directionLabel(human) === directionLabel(model)).length;
  const meanAbsoluteDisagreement = pairs.reduce((total, [human, model]) => total + Math.abs(ORDINAL.get(human) - ORDINAL.get(model)), 0) / pairs.length;
  const severe = pairs.filter(([human, model]) => ORDINAL.get(human) * ORDINAL.get(model) < 0).length;
  return {
    n: pairs.length,
    exact_five_level_agreement: exact / pairs.length,
    directional_agreement: directional / pairs.length,
    weighted_cohens_kappa_quadratic: kappa(pairs, SCALE, (row, column, size) => 1 - ((row - column) / (size - 1)) ** 2),
    collapsed_cohens_kappa: kappa(pairs.map(([human, model]) => [directionLabel(human), directionLabel(model)]), ["A", "Tie", "B"], (row, column) => Number(row === column)),
    mean_absolute_disagreement: meanAbsoluteDisagreement,
    severe_disagreement_rate: severe / pairs.length,
    severe_disagreement_cases: severe,
  };
}

function markdownCell(value) {
  return String(value ?? "").replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}

function analyzeHuman(humanPath) {
  const primaryReceipt = path.join(OUTPUT_DIR, "model_pairwise_primary_freeze.json");
  const reversedReceipt = path.join(OUTPUT_DIR, "model_pairwise_reversed_freeze.json");
  if (!fs.existsSync(primaryReceipt) || !fs.existsSync(reversedReceipt)) {
    throw new Error("both model passes must be frozen before human annotations are read");
  }
  const modelRows = readJsonl(judgmentPath("primary"));
  const modelByCase = new Map(modelRows.map((row) => [row.case_id, row]));
  const humanRows = readJsonl(humanPath);
  const completed = [];
  const seen = new Set();
  for (const row of humanRows) {
    if (!row || typeof row !== "object" || Array.isArray(row)) throw new Error("human export contains a non-object record");
    assertString(row.case_id, "human case_id");
    if (!modelByCase.has(row.case_id)) throw new Error(`human export contains an out-of-sample case: ${row.case_id}`);
    if (seen.has(row.case_id)) throw new Error(`human export contains duplicate case: ${row.case_id}`);
    seen.add(row.case_id);
    completed.push({
      case_id: row.case_id,
      constraint_following: normalizeHumanLabel(row.constraint_following, row.case_id, "constraint_following"),
      creative_quality: normalizeHumanLabel(row.creative_quality, row.case_id, "creative_quality"),
      notes: typeof row.notes === "string" ? row.notes : "",
    });
  }
  if (!completed.length) throw new Error("human export has no completed cases");
  const summary = {
    human_cases_completed: completed.length,
    human_case_ids: completed.map((row) => row.case_id),
    ordinal_encoding: {
      "A clearly better": -2, "A slightly better": -1, Tie: 0,
      "B slightly better": 1, "B clearly better": 2,
    },
    weighting: "quadratic ordinal weights: 1 - ((i - j) / 4)^2 on the ordered five-level scale",
    constraint_following: statistics(completed, modelByCase, "constraint_following", "constraint_following"),
    creative_writing_quality: statistics(completed, modelByCase, "creative_quality", "creative_quality"),
  };
  const disagreements = completed.filter((human) => {
    const model = modelByCase.get(human.case_id);
    return human.constraint_following !== model.constraint_following || human.creative_quality !== model.creative_quality;
  }).map((human) => {
    const model = modelByCase.get(human.case_id);
    return {
      case_id: human.case_id,
      human_constraint_choice: human.constraint_following,
      model_constraint_choice: model.constraint_following,
      human_creative_choice: human.creative_quality,
      model_creative_choice: model.creative_quality,
      human_notes: human.notes,
      model_constraint_reason: model.constraint_reason,
      model_creative_reason: model.creative_reason,
      diagnostic_category: "Not categorized automatically; interpret only from the supplied notes.",
    };
  });
  atomicWrite(path.join(OUTPUT_DIR, "human_model_agreement_summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  const disagreementLines = [
    "# Human–Model Pairwise Disagreements",
    "",
    "Only completed human cases are included. Categories are deliberately not inferred automatically; use the notes as evidence during diagnosis.",
    "",
    "| case_id | human constraint | model constraint | human creative | model creative | human notes | model constraint reason | model creative reason | diagnostic category |",
    "|---|---|---|---|---|---|---|---|---|",
    ...disagreements.map((row) => `| ${markdownCell(row.case_id)} | ${markdownCell(row.human_constraint_choice)} | ${markdownCell(row.model_constraint_choice)} | ${markdownCell(row.human_creative_choice)} | ${markdownCell(row.model_creative_choice)} | ${markdownCell(row.human_notes)} | ${markdownCell(row.model_constraint_reason)} | ${markdownCell(row.model_creative_reason)} | ${markdownCell(row.diagnostic_category)} |`),
    "",
  ];
  atomicWrite(path.join(OUTPUT_DIR, "human_model_disagreements.md"), `${disagreementLines.join("\n")}\n`);
  process.stdout.write(`${JSON.stringify({ completed_human_cases: completed.length, disagreements: disagreements.length })}\n`);
}

function usage() {
  throw new Error("usage: prepare | freeze <primary|reversed> <candidate.jsonl> | report | analyze-human <human-export.jsonl>");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [command, pass, candidatePath] = process.argv.slice(2);
  if (command === "prepare" && !pass) prepare();
  else if (command === "freeze" && PASSES.has(pass) && candidatePath) freeze(pass, path.resolve(candidatePath));
  else if (command === "report" && !pass) report();
  else if (command === "analyze-human" && pass && !candidatePath) analyzeHuman(path.resolve(pass));
  else usage();
}
