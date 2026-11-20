import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "audit_nutrition5k.py"
SPEC = importlib.util.spec_from_file_location("audit_nutrition5k", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AuditNutrition5kTest(unittest.TestCase):
    def test_parse_metadata_rejects_bad_and_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["dish_1", 100, 50, 1, 2, 3])
                writer.writerow(["dish_1", 999, 50, 1, 2, 3])
                writer.writerow(["dish_2", "bad", 50, 1, 2, 3])
                writer.writerow(["dish_3", 100, -1, 1, 2, 3])

            records, audit = MODULE.parse_metadata([path])

        self.assertEqual({"dish_1"}, set(records))
        self.assertEqual(1, audit["metadata_duplicate_ids"])
        self.assertEqual(2, audit["metadata_rows_invalid"])

    def test_selection_is_reproducible_and_can_require_images(self):
        records = [
            MODULE.DishRecord(str(i), 1, 1, 1, 1, 1, "train", f"{i}.png", i % 2 == 0)
            for i in range(10)
        ]
        first = MODULE.select_records(records, 3, 539, True)
        second = MODULE.select_records(records, 3, 539, True)
        self.assertEqual(first, second)
        self.assertTrue(all(record.image_exists for record in first))


if __name__ == "__main__":
    unittest.main()
