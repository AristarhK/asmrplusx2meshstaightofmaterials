"""Synthetic CDB-load counterexamples; importing this module starts no solver."""
import copy
import unittest

from run_mesh_gate import load_check


def one_quad():
    data = {
        "node_ids": [11, 23, 37, 41],
        "xyz_mm": [[0, 0, 0], [1000, 0, 0], [1000, 600, 0], [0, 600, 0]],
        "element_ids": [101],
        "connectivity_node_ids": [[11, 23, 37, 41]],
    }
    commands = [["D", str(n), dof, "0", "0"]
                for n in data["node_ids"]
                for dof in ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")]
    commands.append(["SFE", "101", "1", "PRES", "1", ".005", ".005", ".005", ".005"])
    return data, commands


class LoadChecks(unittest.TestCase):
    def setUp(self):
        self.data, self.commands = one_quad()
        self.config = {"a": 1000., "b": 600., "pressure": .005}

    def check(self):
        return load_check(self.commands, self.data, self.config)

    def test_complete_p1_and_uniform_pressure(self):
        result = self.check()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["boundary_nodes"], 4)
        self.assertEqual(result["prescribed_dof"], 24)
        self.assertEqual(result["expected_free_dof_topology"], 0)
        self.assertEqual(result["loaded_area_mm2"], 600000)
        self.assertEqual(result["pressure_resultant_N"], [0., 0., 3000.])
        self.assertIsNone(result["solver_equation_count"])

    def test_missing_rotation_constraint(self):
        self.commands.pop(5)
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["exact_p1_dof_set"])

    def test_extra_interior_constraint(self):
        self.data["node_ids"].append(97)
        self.data["xyz_mm"].append([500, 300, 0])
        self.commands.append(["D", "97", "UZ", "0", "0"])
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["exact_p1_dof_set"])

    def test_duplicate_constraint_record(self):
        self.commands.append(copy.deepcopy(self.commands[0]))
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["no_duplicate_dof_records"])

    def test_nonzero_imaginary_displacement(self):
        self.commands[0][4] = "1D-3"
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["zero_imaginary_displacements"])

    def test_missing_pressure(self):
        self.commands.pop()
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_elements_face1_pressure_correct"])

    def test_wrong_face(self):
        self.commands[-1][2] = "2"
        self.assertFalse(self.check()["passed"])

    def test_one_wrong_corner_pressure(self):
        self.commands[-1][-1] = ".004"
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_elements_face1_pressure_correct"])

    def test_incomplete_four_corner_pressure(self):
        self.commands[-1][-1] = ""
        self.assertFalse(self.check()["passed"])

    def test_duplicate_pressure_record(self):
        self.commands.append(copy.deepcopy(self.commands[-1]))
        self.assertFalse(self.check()["passed"])

    def test_unexpected_load_or_constraint(self):
        self.commands.append(["F", "11", "FZ", "10"])
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["other_load_or_constraint_commands"], ["F"])


if __name__ == "__main__":
    unittest.main()
