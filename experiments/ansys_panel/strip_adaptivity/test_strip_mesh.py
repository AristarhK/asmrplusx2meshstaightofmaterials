"""Deterministic strip topology/marking/budget tests; no solver interface."""
from pathlib import Path
import sys
import unittest

import numpy as np

from strip_mesh import (DOF_CAP, build_mesh, classical_selector, counts,
                        dorfler_mark, maximum_aspect, remesh)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"meshing"))
from validate_mesh import validate_mesh


CONFIG = {"a": 1000., "b": 600., "thickness": 8.,
          "keyopts": {"1":0,"3":2,"8":2}, "thickness_integration_points": 5}


def validate(mesh, config):
    return validate_mesh(mesh["node_ids"], mesh["xyz_mm"], mesh["element_ids"],
                         mesh["connectivity_node_ids"], mesh["attributes"], config)


class StripTopology(unittest.TestCase):
    def test_original_panel_is_conforming_positive_z_and_full_area(self):
        mesh = build_mesh(np.arange(21)*50., np.arange(13)*50., CONFIG)
        self.assertEqual(mesh["connectivity_node_ids"][0], [1,2,23,22])
        self.assertEqual(mesh["element_ids"], list(range(1,241)))
        self.assertEqual(mesh["node_ids"], list(range(1,274)))
        self.assertEqual(mesh["topology_prediction"]["expected_free_dof_topology"], 1254)
        result = validate(mesh, CONFIG)
        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["metrics"]["signed_area_mm2"], 600000.)
        self.assertEqual(result["metrics"]["normal_z_minimum"], 1.)
        self.assertEqual(result["metrics"]["node_on_foreign_edge_count"], 0)
        self.assertIn("Requested", mesh["native_settings_provenance"])
        self.assertIsNone(mesh["topology_prediction"]["solver_equation_count"])
        self.assertEqual(mesh["native_settings"]["keyopts"], {"1":0,"3":2,"8":2})

    def test_uniform_h25_exact_budget(self):
        x, y = np.arange(41)*25., np.arange(25)*25.
        result = counts(x,y)
        self.assertEqual(result["elements"], 960)
        self.assertEqual(result["expected_free_dof_topology"], DOF_CAP)
        self.assertEqual(result["expected_free_dof_topology"], 5382)

    def test_marked_square_tie_chooses_x_and_deduplicates_same_column(self):
        x, y = np.arange(21)*50., np.arange(13)*50.
        saved_x, saved_y = x.copy(), y.copy()
        new_x, new_y, audit = remesh(x,y,[21,1],CONFIG)
        self.assertEqual(audit["marked_element_ids"], [1,21])
        self.assertEqual(audit["requested_new_line_count"], 1)
        self.assertEqual(audit["requested_interval_splits"][0]["axis"], "X")
        self.assertEqual(audit["requested_interval_splits"][0]["source_marked_element_ids"], [1,21])
        self.assertEqual(audit["requested_interval_splits"][0]["midpoint_mm"], 25.)
        np.testing.assert_array_equal(new_y,y)
        np.testing.assert_array_equal(x,saved_x); np.testing.assert_array_equal(y,saved_y)
        self.assertTrue(set(x).issubset(set(new_x)))
        self.assertTrue(np.all(np.diff(new_x)>0))
        self.assertEqual(audit["closure_new_line_count"], 0)
        self.assertEqual(audit["after"]["expected_free_dof_topology"], 1320)

    def test_longer_y_interval_is_chosen(self):
        config = dict(CONFIG,a=50.,b=100.)
        x,y,audit = remesh([0,25,50],[0,50,100],[1],config)
        np.testing.assert_array_equal(x,[0,25,50])
        np.testing.assert_array_equal(y,[0,25,50,100])
        self.assertEqual(audit["cell_split_choices"][0]["requested_axis"],"Y")

    def test_closure_uses_largest_offender_then_axis_then_current_index(self):
        config = dict(CONFIG,a=9.,b=9.)
        x,y,audit = remesh([0,1,9],[0,1,9],[1],config)
        first = audit["closure_splits"][0]
        self.assertEqual(first["axis"],"X")
        self.assertEqual((first["left_mm"],first["right_mm"]),(1.,9.))
        self.assertEqual(audit["closure_splits"][1]["axis"],"Y")
        self.assertLessEqual(maximum_aspect(x,y),4.)
        self.assertEqual(audit["total_new_line_count"],len(x)+len(y)-6)
        result = validate(build_mesh(x,y,config), config)
        self.assertTrue(result["passed"],result["failures"])

    def test_closure_same_axis_length_tie_selects_lowest_current_index(self):
        config = dict(CONFIG,a=8.,b=1.)
        x,y,audit = remesh([0,8],[0,.25,1],[2],config)
        first,second = audit["closure_splits"][:2]
        self.assertEqual((first["left_mm"],first["right_mm"]),(0.,4.))
        self.assertEqual((second["left_mm"],second["right_mm"]),(4.,8.))
        self.assertLessEqual(maximum_aspect(x,y),4.)
        self.assertEqual(audit["closure_new_line_count"],6)

    def test_two_full_mark_steps_reach_exact_cap_then_reject_whole_proposal(self):
        x,y = np.arange(21)*50., np.arange(13)*50.
        x,y,first = remesh(x,y,np.arange(1,241),CONFIG)
        self.assertEqual((first["after"]["nx"],first["after"]["ny"]),(40,12))
        self.assertEqual(first["after"]["expected_free_dof_topology"],2574)
        x,y,second = remesh(x,y,np.arange(1,481),CONFIG)
        self.assertEqual((second["after"]["nx"],second["after"]["ny"]),(40,24))
        self.assertEqual(second["after"]["expected_free_dof_topology"],5382)
        self.assertTrue(second["within_dof_budget"])
        self.assertFalse(second["rejected_before_solve"])
        proposed_x,proposed_y,third = remesh(x,y,np.arange(1,961),CONFIG)
        self.assertEqual((third["after"]["nx"],third["after"]["ny"]),(80,24))
        self.assertEqual(third["requested_new_line_count"],40)
        self.assertEqual(third["after"]["expected_free_dof_topology"],10902)
        self.assertFalse(third["within_dof_budget"])
        self.assertTrue(third["rejected_before_solve"])
        self.assertEqual(len(proposed_x),81); self.assertEqual(len(proposed_y),25)
        self.assertFalse(third["solve_performed"])

    def test_invalid_axes_and_marks_are_rejected(self):
        for x,y in (([0,1,1,1000],[0,600]),([1,1000],[0,600]),([0,1000],[0,np.nan,600])):
            with self.subTest(x=x,y=y), self.assertRaises(ValueError):
                build_mesh(x,y,CONFIG)
        for marks in ([],[1,1],[0],[241],[1.5]):
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                remesh(np.arange(21)*50.,np.arange(13)*50.,marks,CONFIG)


class Marking(unittest.TestCase):
    def test_exact_score_ties_use_element_id(self):
        selected,audit = dorfler_mark([7,3,10,1],[1,1,1,1])
        self.assertEqual(selected,[1,3])
        self.assertEqual(audit["ordered_element_ids"],[1,3,7,10])
        self.assertEqual(audit["selected_fraction"],.5)

    def test_minimal_prefix_and_selector_preserve_eta_squared(self):
        scores = np.array([4.,3.,2.,1.])
        observation={"element_ids":[11,17,29,31],"eta2":scores}
        values=classical_selector(observation)
        self.assertFalse(np.shares_memory(values,scores))
        np.testing.assert_array_equal(values,scores)
        selected,audit=dorfler_mark(observation["element_ids"],values)
        self.assertEqual(selected,[11,17])
        self.assertLess(4.,audit["target_eta2"])
        self.assertGreaterEqual(audit["selected_eta2"],audit["target_eta2"])

    def test_zero_or_invalid_scores_fail_without_fallback(self):
        for values in ([0,0],[-1,2],[float("nan"),1],[float("inf"),1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                dorfler_mark([1,2],values)
        with self.assertRaisesRegex(ValueError,"fixes"):
            dorfler_mark([1,2],[1,2],theta=.8)
        with self.assertRaises(ValueError):
            dorfler_mark([1,1],[1,1])


if __name__ == "__main__":
    unittest.main()
