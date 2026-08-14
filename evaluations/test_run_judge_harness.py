import copy
import hashlib
import unittest

import run_judge_harness as harness


BATCH = [
    {
        "record_id": "record-1",
        "story_id": "fantasy_001",
        "full_instruction": "Write a fantasy story.",
        "constraints": [
            {"constraint_id": "fantasy_001_c1", "requirement": "Use fantasy."},
            {"constraint_id": "fantasy_001_c2", "requirement": "Include a dragon."},
        ],
        "text": "A dragon guarded the fantasy kingdom.",
    }
]

VALID_RESPONSE = {
    "evaluations": [
        {
            "record_id": "record-1",
            "adherence": [
                {"constraint_id": "fantasy_001_c1", "score": 1, "reason": "It is fantasy."},
                {"constraint_id": "fantasy_001_c2", "score": 0.5, "reason": "A dragon appears briefly."},
            ],
            "quality": {
                "craft": 3,
                "structure_coherence": 3,
                "originality": 3,
                "genre_effectiveness": 3,
                "characterization": 3,
                "characterization_na": False,
            },
        }
    ]
}


class JudgeResponseValidationTests(unittest.TestCase):
    def test_validate_response_derives_stable_id_and_instruction_mean(self):
        rows = harness.validate_response(BATCH, VALID_RESPONSE, "v1")

        self.assertEqual(
            rows[0]["evaluation_id"],
            hashlib.sha256(b"record-1v1").hexdigest(),
        )
        self.assertEqual(rows[0]["I_i"], 0.75)
        self.assertEqual(rows[0]["quality"]["characterization"], 3)

    def test_validate_response_rejects_missing_constraint_score(self):
        invalid = copy.deepcopy(VALID_RESPONSE)
        invalid["evaluations"][0]["adherence"].pop()

        with self.assertRaisesRegex(ValueError, "constraint"):
            harness.validate_response(BATCH, invalid, "v1")

    def test_validate_response_rejects_inconsistent_nullable_characterization(self):
        invalid = copy.deepcopy(VALID_RESPONSE)
        invalid["evaluations"][0]["quality"]["characterization"] = None
        invalid["evaluations"][0]["quality"]["characterization_na"] = False

        with self.assertRaisesRegex(ValueError, "characterization"):
            harness.validate_response(BATCH, invalid, "v1")


if __name__ == "__main__":
    unittest.main()
