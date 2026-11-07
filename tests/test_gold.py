import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from build_gold import check_constraints, map_type_group, select_negatives


class TestGold(unittest.TestCase):
    def test_selection_order_and_quota(self):
        # Selection takes the first COMPATIBLE items in order_index order, up to the quota
        candidates = [
            {"cand_id": 1, "split": "dev", "order_index": 30, "article_title": "A", "passage1": "p1", "passage2": "p2"},
            {"cand_id": 2, "split": "dev", "order_index": 10, "article_title": "B", "passage1": "p1", "passage2": "p2"},
            {"cand_id": 3, "split": "dev", "order_index": 20, "article_title": "C", "passage1": "p1", "passage2": "p2"},
            {"cand_id": 4, "split": "dev", "order_index": 5, "article_title": "D", "passage1": "p1", "passage2": "p2"},
        ]
        labels = {
            "1": {"label": "COMPATIBLE"},
            "2": {"label": "COMPATIBLE"},
            "3": {"label": "COMPATIBLE"},
            "4": {"label": "CONTRADICT"},
        }
        res = select_negatives(candidates, labels, exclude_ids=set(), quotas={"dev": 2, "test": 0})
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["id"], "neg_2")
        self.assertEqual(res[1]["id"], "neg_3")

    def test_excluded_id_dropped_and_not_replaced(self):
        # An excluded id is dropped and not replaced
        candidates = [
            {"cand_id": 1, "split": "dev", "order_index": 1, "article_title": "A", "passage1": "p1", "passage2": "p2"},
            {"cand_id": 2, "split": "dev", "order_index": 2, "article_title": "B", "passage1": "p1", "passage2": "p2"},
        ]
        labels = {
            "1": {"label": "COMPATIBLE"},
            "2": {"label": "COMPATIBLE"},
        }
        res = select_negatives(candidates, labels, exclude_ids={"1"}, quotas={"dev": 2, "test": 0})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["id"], "neg_2")

    def test_edited_negative_uses_edited_text(self):
        # An edited negative uses its edited text
        candidates = [
            {"cand_id": 1, "split": "dev", "order_index": 1, "article_title": "A", "passage1": "orig1", "passage2": "orig2"}
        ]
        labels = {
            "1": {
                "label": "COMPATIBLE",
                "edited": True,
                "original_text": {"passage1": "orig1", "passage2": "orig2"},
                "edited_text": {"passage1": "edited1", "passage2": "edited2"},
            }
        }
        res = select_negatives(candidates, labels, exclude_ids=set(), quotas={"dev": 1, "test": 0})
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["passage1"], "edited1")
        self.assertEqual(res[0]["passage2"], "edited2")
        self.assertTrue(res[0]["edited"])

    def test_article_overlap_check_raises(self):
        # The article-overlap check raises an error on a split-overlap example
        items = [
            {"id": "pos_1", "split": "dev", "article": "Shared Article", "passage1": "A1", "passage2": "A2", "label": 1},
            {"id": "neg_1", "split": "test", "article": "Shared Article", "passage1": "B1", "passage2": "B2", "label": 0},
        ]
        with self.assertRaises(ValueError):
            check_constraints(items)

    def test_type_group_mapping(self):
        # The type_group mapping covers date_time, number, other and unknown
        self.assertEqual(map_type_group("(PhraseLevel) Entity - Date/time"), "date_time")
        self.assertEqual(map_type_group("(PhraseLevel) Entity - Number"), "number")
        self.assertEqual(map_type_group("(PhraseLevel) Entity - Location"), "other")
        self.assertEqual(map_type_group("(DiscourseLevel) Event/Relation-related"), "other")
        self.assertEqual(map_type_group(None), "unknown")
        self.assertEqual(map_type_group(""), "unknown")


if __name__ == "__main__":
    unittest.main()

