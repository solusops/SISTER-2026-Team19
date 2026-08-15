import json
import tempfile
import unittest
from pathlib import Path

from merge_scores import load_constraints, load_final_record_ids, load_sources, validate


def quality(**overrides):
    base = {
        "craft": 3,
        "structure_coherence": 3,
        "originality": 3,
        "genre_effectiveness": 3,
        "characterization": 3,
        "characterization_na": False,
    }
    base.update(overrides)
    return base


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def make_eval_dir(root: Path, *, auto=(), native=(), long=(), repair=()) -> Path:
    eval_dir = root / "evaluations"
    write_jsonl(eval_dir / "scores_auto_nonoverlap.jsonl", list(auto))
    write_jsonl(eval_dir / "native_judge_scores" / "batch-001.jsonl", list(native))
    write_jsonl(eval_dir / "native_judge_long_scores" / "long-001.jsonl", list(long))
    write_jsonl(eval_dir / "native_judge_repair_scores" / "repair-001.jsonl", list(repair))
    return eval_dir


def constraints_row(story_id, n=2):
    return {
        "story_id": story_id,
        "constraints": [{"constraint_id": f"{story_id}_c{i}"} for i in range(1, n + 1)],
    }


def adherence_for(story_id, n=2):
    return [{"constraint_id": f"{story_id}_c{i}", "score": 1, "reason": "ok"} for i in range(1, n + 1)]


class LoadSourcesTests(unittest.TestCase):
    def test_repair_rows_supersede_native_and_long(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_defective = {
                "record_id": "rec-native-bad",
                "story_id": "comedy_001",
                "adherence": adherence_for("comedy_001"),
                "quality": quality(),
                "source": "codex_native_terra_medium",
            }
            repair_row = {
                "record_id": "rec-native-bad",
                "story_id": "comedy_001",
                "adherence": adherence_for("comedy_001"),
                "quality": quality(),
                "source": "codex_native_sol_high_repair",
            }
            eval_dir = make_eval_dir(root, native=[native_defective], repair=[repair_row])

            rows, counts = load_sources(eval_dir)

            self.assertEqual(counts, {"auto_nonoverlap": 0, "native": 0, "long": 0, "repair": 1})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source"], "codex_native_sol_high_repair")

    def test_auto_rows_get_story_id_derived_from_constraint_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            auto_row = {
                "record_id": "rec-auto",
                "adherence": adherence_for("historical_fiction_005"),
                "quality": quality(),
                "source": "auto_judge_sonnet_medium_effort",
            }
            eval_dir = make_eval_dir(root, auto=[auto_row])

            rows, _ = load_sources(eval_dir)

            self.assertEqual(rows[0]["story_id"], "historical_fiction_005")


class ValidateTests(unittest.TestCase):
    def setUp(self):
        self.constraints = {
            "comedy_001": {c["constraint_id"] for c in constraints_row("comedy_001")["constraints"]}
        }
        self.final_ids = {"rec-1"}

    def _row(self, **overrides):
        row = {
            "record_id": "rec-1",
            "story_id": "comedy_001",
            "adherence": adherence_for("comedy_001"),
            "quality": quality(),
            "source": "auto_judge_sonnet_medium_effort",
        }
        row.update(overrides)
        return row

    def test_valid_row_passes(self):
        validate([self._row()], self.constraints, self.final_ids)

    def test_rejects_duplicate_record_id(self):
        with self.assertRaises(SystemExit) as ctx:
            validate([self._row(), self._row()], self.constraints, self.final_ids)
        self.assertIn("duplicate record_id", str(ctx.exception))

    def test_rejects_record_not_in_final_generations(self):
        with self.assertRaises(SystemExit) as ctx:
            validate([self._row(record_id="ghost")], self.constraints, self.final_ids)
        self.assertIn("not a final generation record", str(ctx.exception))

    def test_rejects_constraint_id_mismatch(self):
        bad = self._row(adherence=adherence_for("comedy_001", n=1))
        with self.assertRaises(SystemExit) as ctx:
            validate([bad], self.constraints, self.final_ids)
        self.assertIn("constraint-ID mismatch", str(ctx.exception))

    def test_rejects_non_canonical_quality_keys(self):
        bad = self._row(quality=quality())
        bad["quality"]["prose_craft"] = bad["quality"].pop("craft")
        with self.assertRaises(SystemExit) as ctx:
            validate([bad], self.constraints, self.final_ids)
        self.assertIn("quality keys", str(ctx.exception))

    def test_rejects_out_of_scale_adherence_score(self):
        bad = self._row(adherence=[{"constraint_id": "comedy_001_c1", "score": 0.75, "reason": "x"},
                                    {"constraint_id": "comedy_001_c2", "score": 1, "reason": "x"}])
        with self.assertRaises(SystemExit) as ctx:
            validate([bad], self.constraints, self.final_ids)
        self.assertIn("out of scale", str(ctx.exception))

    def test_rejects_characterization_na_mismatch(self):
        bad = self._row(quality=quality(characterization_na=True, characterization=3))
        with self.assertRaises(SystemExit) as ctx:
            validate([bad], self.constraints, self.final_ids)
        self.assertIn("characterization_na=True", str(ctx.exception))


class LoadConstraintsTests(unittest.TestCase):
    def test_builds_story_id_to_constraint_id_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "constraints.jsonl"
            write_jsonl(path, [constraints_row("comedy_001", n=3)])
            self.assertEqual(
                load_constraints(path),
                {"comedy_001": {"comedy_001_c1", "comedy_001_c2", "comedy_001_c3"}},
            )


class LoadFinalRecordIdsTests(unittest.TestCase):
    def test_only_final_records_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "all_results.jsonl"
            write_jsonl(path, [
                {"record_id": "final-1", "is_final": True},
                {"record_id": "draft-1", "is_final": False},
            ])
            self.assertEqual(load_final_record_ids(path), {"final-1"})


if __name__ == "__main__":
    unittest.main()
