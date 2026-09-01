/**
 * Validates, freezes, and analyzes evidence-first pairwise judgments for the
 * fixed 30-case blind sample. Judge contexts receive only sanitized cases and
 * the rubric; this utility never reads human or earlier judge judgments.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const DIR = path.join(ROOT, "evaluations", "human_validation");
const PASSES = new Set(["primary", "reversed"]);
const SCALE = ["A clearly better", "A slightly better", "Tie", "B slightly better", "B clearly better"];
const SCALE_SET = new Set(SCALE);
const STATUS_SET = new Set(["satisfied", "partial", "violated"]);
const DIMENSIONS = ["prose_and_craft", "coherence_and_progression", "originality_of_execution", "natural_constraint_integration", "characterization", "atmosphere_and_effect", "narrative_payoff", "genre_effectiveness"];
const JUDGMENT_FIELDS = new Set(["case_id", "constraint_analysis", "constraint_following", "constraint_summary", "creative_dimensions", "creative_quality", "creative_summary"]);
const CONSTRAINT_FIELDS = new Set(["constraint_id", "requirement", "A", "B"]);
const RESPONSE_EVIDENCE_FIELDS = new Set(["status", "evidence"]);
const DIMENSION_FIELDS = new Set(["winner", "evidence_A", "evidence_B", "comparison"]);

function sha256(text) { return crypto.createHash("sha256").update(text, "utf8").digest("hex"); }
function readJsonl(file) { return fs.readFileSync(file, "utf8").split(/\r?\n/).filter(Boolean).map(JSON.parse); }
function atomicWrite(file, text) {
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${process.pid}.${Date.now()}.tmp`);
  fs.writeFileSync(temporary, text, "utf8");
  fs.renameSync(temporary, file);
}
function assertString(value, label) { if (typeof value !== "string" || !value.trim()) throw new Error(`${label} must be a non-empty string`); }
function assertFields(value, fields, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object`);
  for (const key of Object.keys(value)) if (!fields.has(key)) throw new Error(`${label} contains unexpected field ${key}`);
  for (const key of fields) if (!(key in value)) throw new Error(`${label} is missing ${key}`);
}
function casePath(pass) { return path.join(DIR, `sanitized_${pass}_cases.jsonl`); }
function outputPath(pass) { return path.join(DIR, `evidence_first_${pass}.jsonl`); }

function validateJudgments(cases, judgments) {
  if (cases.length !== 30 || judgments.length !== 30) throw new Error("exactly 30 input cases and 30 judgments are required");
  const byId = new Map(cases.map((item) => [item.case_id, item]));
  const seen = new Set();
  for (const judgment of judgments) {
    assertFields(judgment, JUDGMENT_FIELDS, "judgment");
    assertString(judgment.case_id, "case_id");
    const source = byId.get(judgment.case_id);
    if (!source) throw new Error(`unexpected case ${judgment.case_id}`);
    if (seen.has(judgment.case_id)) throw new Error(`duplicate judgment ${judgment.case_id}`);
    seen.add(judgment.case_id);
    if (!Array.isArray(judgment.constraint_analysis) || judgment.constraint_analysis.length !== source.constraints.length) throw new Error(`${judgment.case_id} has incomplete constraint analysis`);
    const expectedConstraints = new Map(source.constraints.map((constraint) => [constraint.constraint_id, constraint]));
    const audited = new Set();
    for (const item of judgment.constraint_analysis) {
      assertFields(item, CONSTRAINT_FIELDS, `${judgment.case_id} constraint`);
      const expected = expectedConstraints.get(item.constraint_id);
      if (!expected || audited.has(item.constraint_id)) throw new Error(`${judgment.case_id} has incorrect constraint IDs`);
      audited.add(item.constraint_id);
      if (item.requirement !== expected.requirement) throw new Error(`${judgment.case_id} changed requirement text for ${item.constraint_id}`);
      for (const side of ["A", "B"]) {
        assertFields(item[side], RESPONSE_EVIDENCE_FIELDS, `${judgment.case_id} ${item.constraint_id} ${side}`);
        if (!STATUS_SET.has(item[side].status)) throw new Error(`${judgment.case_id} has invalid constraint status`);
        assertString(item[side].evidence, `${judgment.case_id} ${item.constraint_id} ${side} evidence`);
      }
    }
    if (!SCALE_SET.has(judgment.constraint_following) || !SCALE_SET.has(judgment.creative_quality)) throw new Error(`${judgment.case_id} has invalid final preference`);
    assertString(judgment.constraint_summary, `${judgment.case_id} constraint_summary`);
    assertString(judgment.creative_summary, `${judgment.case_id} creative_summary`);
    assertFields(judgment.creative_dimensions, new Set(DIMENSIONS), `${judgment.case_id} creative_dimensions`);
    for (const dimension of DIMENSIONS) {
      const item = judgment.creative_dimensions[dimension];
      assertFields(item, DIMENSION_FIELDS, `${judgment.case_id} ${dimension}`);
      if (!["A", "Tie", "B"].includes(item.winner) && !(dimension === "characterization" && item.winner === "not_applicable")) throw new Error(`${judgment.case_id} has invalid ${dimension} winner`);
      assertString(item.evidence_A, `${judgment.case_id} ${dimension} evidence_A`);
      assertString(item.evidence_B, `${judgment.case_id} ${dimension} evidence_B`);
      assertString(item.comparison, `${judgment.case_id} ${dimension} comparison`);
    }
  }
  if (seen.size !== 30) throw new Error("missing judgments");
  return cases.map((item) => judgments.find((judgment) => judgment.case_id === item.case_id));
}

function freeze(pass, candidate) {
  if (!PASSES.has(pass)) throw new Error("pass must be primary or reversed");
  const cases = readJsonl(casePath(pass));
  const frozen = validateJudgments(cases, readJsonl(candidate));
  const serialized = `${frozen.map((item) => JSON.stringify(item)).join("\n")}\n`;
  atomicWrite(outputPath(pass), serialized);
  atomicWrite(path.join(DIR, `evidence_first_${pass}_freeze.json`), `${JSON.stringify({
    pass, cases: 30, responses: 60, input_sha256: sha256(fs.readFileSync(casePath(pass), "utf8")), output_sha256: sha256(serialized), human_or_prior_judgment_comparison_performed: false,
  }, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ pass, cases: 30, responses: 60, sha256: sha256(serialized) })}\n`);
}

function assemble(pass, candidateDirectory) {
  if (!PASSES.has(pass)) throw new Error("pass must be primary or reversed");
  const files = fs.readdirSync(candidateDirectory).filter((file) => file.endsWith(".jsonl")).sort();
  if (!files.length) throw new Error("candidate directory has no JSONL files");
  const candidate = files.flatMap((file) => readJsonl(path.join(candidateDirectory, file)));
  const ordered = validateJudgments(readJsonl(casePath(pass)), candidate);
  const output = path.join(candidateDirectory, `evidence_first_${pass}_candidate.jsonl`);
  atomicWrite(output, `${ordered.map((item) => JSON.stringify(item)).join("\n")}\n`);
  process.stdout.write(`${JSON.stringify({ pass, batches: files.length, cases: ordered.length, candidate: output })}\n`);
}

function inversePreference(label) {
  const index = SCALE.indexOf(label);
  if (index < 0) throw new Error(`unknown preference ${label}`);
  return SCALE[4 - index];
}
function inverseWinner(winner) { return winner === "A" ? "B" : winner === "B" ? "A" : winner; }
function sign(label) { return Math.sign(SCALE.indexOf(label) - 2); }
function pairRows() {
  const primary = readJsonl(outputPath("primary"));
  const reversed = new Map(readJsonl(outputPath("reversed")).map((item) => [item.case_id, item]));
  if (primary.length !== 30 || reversed.size !== 30) throw new Error("both frozen evidence-first files must have 30 cases");
  return primary.map((item) => [item, reversed.get(item.case_id)]);
}

function analyze() {
  const pairs = pairRows();
  const final = {};
  const finalDiagnostics = [];
  for (const field of ["constraint_following", "creative_quality"]) {
    let exact = 0; let directional = 0; let opposite = 0;
    for (const [primary, reversed] of pairs) {
      const restored = inversePreference(reversed[field]);
      if (primary[field] === restored) exact += 1;
      if (sign(primary[field]) === sign(restored)) directional += 1;
      if (sign(primary[field]) * sign(restored) < 0) opposite += 1;
      if (primary[field] !== restored) {
        const evidenceChanged = field === "constraint_following"
          ? primary.constraint_analysis.some((item) => {
            const counterpart = reversed.constraint_analysis.find((candidate) => candidate.constraint_id === item.constraint_id);
            return item.A.status !== counterpart.B.status || item.B.status !== counterpart.A.status;
          })
          : DIMENSIONS.some((dimension) => primary.creative_dimensions[dimension].winner !== inverseWinner(reversed.creative_dimensions[dimension].winner));
        finalDiagnostics.push({ case_id: primary.case_id, field, primary: primary[field], reversed_restored: restored, evidence_decisions_changed: evidenceChanged, failure_mode: evidenceChanged ? "underlying evidence decision changed" : "final aggregation changed while evidence decisions remained stable" });
      }
    }
    final[field] = { n: 30, exact_five_level_consistency: exact / 30, directional_consistency: directional / 30, opposite_direction_reversals: opposite };
  }
  const constraintEntries = []; let stableConstraints = 0;
  for (const [primary, reversed] of pairs) {
    for (const item of primary.constraint_analysis) {
      const counterpart = reversed.constraint_analysis.find((candidate) => candidate.constraint_id === item.constraint_id);
      for (const [side, reversedSide] of [["A", "B"], ["B", "A"]]) {
        const stable = item[side].status === counterpart[reversedSide].status;
        if (stable) stableConstraints += 1;
        constraintEntries.push({ case_id: primary.case_id, constraint_id: item.constraint_id, response_in_primary_orientation: side, primary_status: item[side].status, reversed_restored_status: counterpart[reversedSide].status, stable });
      }
    }
  }
  const dimensionRows = []; const dimensionSummary = {};
  for (const dimension of DIMENSIONS) {
    let stable = 0;
    for (const [primary, reversed] of pairs) {
      const restored = inverseWinner(reversed.creative_dimensions[dimension].winner);
      const isStable = primary.creative_dimensions[dimension].winner === restored;
      if (isStable) stable += 1;
      dimensionRows.push({ case_id: primary.case_id, dimension, primary_winner: primary.creative_dimensions[dimension].winner, reversed_restored_winner: restored, stable: isStable });
    }
    dimensionSummary[dimension] = { n: 30, winner_consistency: stable / 30, unstable_cases: 30 - stable };
  }
  const position = { orientation: "Reversed A/B labels are restored to the primary orientation before comparison.", final_preference_stability: final, final_preference_change_diagnostics: finalDiagnostics };
  const constraints = { audited_response_constraint_decisions: constraintEntries.length, stable_decisions: stableConstraints, constraint_status_consistency: stableConstraints / constraintEntries.length, entries: constraintEntries };
  const dimensions = { per_dimension: dimensionSummary, entries: dimensionRows };
  atomicWrite(path.join(DIR, "evidence_first_position_consistency.json"), `${JSON.stringify(position, null, 2)}\n`);
  atomicWrite(path.join(DIR, "evidence_first_constraint_stability.json"), `${JSON.stringify(constraints, null, 2)}\n`);
  atomicWrite(path.join(DIR, "evidence_first_dimension_stability.json"), `${JSON.stringify(dimensions, null, 2)}\n`);
  const lines = [
    "# Evidence-First Pairwise Judge Validation", "",
    "## Procedure", "",
    "The judge completed a compact structured evidence audit before each final preference: per-constraint satisfied/partial/violated status for both responses, then eight grounded creative dimensions. These are inspectable evidence-based intermediate judgments, not hidden chain-of-thought.",
    "Both primary and A/B-reversed passes were judged in fresh contexts using only sanitized cases and the rubric. Neither pass received human annotations, prior pairwise outputs, pointwise scores, provenance, or position-consistency results.", "",
    "## Final-preference robustness", "",
    `- Constraint following: exact ${(final.constraint_following.exact_five_level_consistency * 100).toFixed(1)}%; directional ${(final.constraint_following.directional_consistency * 100).toFixed(1)}%; opposite-direction reversals ${final.constraint_following.opposite_direction_reversals}/30.`,
    `- Creative quality: exact ${(final.creative_quality.exact_five_level_consistency * 100).toFixed(1)}%; directional ${(final.creative_quality.directional_consistency * 100).toFixed(1)}%; opposite-direction reversals ${final.creative_quality.opposite_direction_reversals}/30.`, "",
    "## Evidence-level robustness", "",
    `- Constraint status decisions stable after restoring orientation: ${stableConstraints}/${constraintEntries.length} (${(stableConstraints / constraintEntries.length * 100).toFixed(1)}%).`,
    ...DIMENSIONS.map((dimension) => `- ${dimension}: ${(dimensionSummary[dimension].winner_consistency * 100).toFixed(1)}% dimension-winner consistency.`), "",
    "## Interpretation and limitations", "",
    "A changed final preference can arise from changed evidence decisions or from changed final aggregation despite stable evidence decisions; `evidence_first_position_consistency.json` separates these cases. This study measures order robustness and inspectability on the fixed N=30 sample, not agreement with humans or the full 1,920-output dataset. No human comparison has been performed for the evidence-first evaluator.", "",
  ];
  atomicWrite(path.join(DIR, "evidence_first_methodology.md"), `${lines.join("\n")}\n`);
  process.stdout.write(`${JSON.stringify({ final, constraint_status_consistency: constraints.constraint_status_consistency, dimensions: dimensionSummary })}\n`);
}

const [command, pass, candidate] = process.argv.slice(2);
if (command === "freeze" && PASSES.has(pass) && candidate) freeze(pass, path.resolve(candidate));
else if (command === "assemble" && PASSES.has(pass) && candidate) assemble(pass, path.resolve(candidate));
else if (command === "analyze" && !pass) analyze();
else throw new Error("usage: assemble <primary|reversed> <candidate-dir> | freeze <primary|reversed> <candidate.jsonl> | analyze");
