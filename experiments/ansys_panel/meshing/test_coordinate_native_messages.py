"""Regressions for unloaded/loaded checks and repeated native shape messages."""
import unittest
from check_optimized_native import classify_messages


class NativeMessages(unittest.TestCase):
    def test_unloaded_no_d_warning_is_preserved_and_expected(self):
        items=[{"stage":"check_unloaded","severity":"WARNING","message":"No constraints have been defined using the D command."}]
        result=classify_messages(items)
        self.assertEqual(result["expected_unloaded_no_constraints_warning_count"],1)
        self.assertEqual(result["unexpected_warning_count"],0)
        self.assertTrue(items[0]["expected_unloaded_no_constraints_warning"])

    def test_no_d_warning_after_loading_remains_failure(self):
        result=classify_messages([{"stage":"check_loaded","severity":"WARNING","message":"No constraints have been defined using the D command."}])
        self.assertEqual(result["unexpected_warning_count"],1)
        self.assertEqual(result["nonshape_unexpected_warning_count"],1)

    def test_shape_warning_messages_are_separate_from_nonshape_count(self):
        body="Quadrilateral element 245 has a pair of opposite edges that are 79.07\ndegrees away from being parallel.  This exceeds the warning limit of\n70 degrees."
        items=[{"stage":stage,"severity":"WARNING","message":body} for stage in ("build","check_unloaded","check_loaded")]
        result=classify_messages(items)
        self.assertEqual(result["unexpected_warning_count"],3)
        self.assertEqual(result["nonshape_unexpected_warning_count"],0)
        self.assertTrue(all(item["shape_diagnostic"] for item in items))

    def test_unknown_warning_and_errors_are_never_suppressed(self):
        result=classify_messages([{"stage":"check_loaded","severity":"WARNING","message":"An unfamiliar diagnostic"},{"stage":"build","severity":"ERROR","message":"Element failure"}])
        self.assertEqual(result["unexpected_warning_count"],1)
        self.assertEqual(result["nonshape_unexpected_warning_count"],1)
        self.assertEqual(result["error_or_fatal_count"],1)


if __name__=="__main__":unittest.main()
