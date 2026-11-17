"""Offline runner contracts; no session, native solve, or fabricated result set."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_experiment import ERROR_KEYS, assess, bilinear_stress, execute_case
from recovery import build_observation
from strip_mesh import build_mesh, classical_selector, dorfler_mark


CONFIG = {"a": 1000., "b": 600., "thickness": 8., "young": 210000.,
          "poisson": .3, "pressure": .005,
          "keyopts": {"1":0,"3":2,"8":2}, "thickness_integration_points": 5}


def assessment_reference():
    result = {"central_deflection": 500., "disks": {"top": {}, "bottom": {}},
              "diagnostic_max_VM": {"top": 999999., "bottom": 777777.}}
    for index, (symbol, name) in enumerate(zip(("C","Y","X"), ("center","near_y_edge","near_x_edge"))):
        for side, surface in enumerate(("top","bottom")):
            result["disks"][surface][name] = {"mean_von_mises": float(100*2**(2*index+side))}
    return result


def change_metric(record, key, delta):
    if key == "w":
        record["central_deflection"] += delta
    else:
        symbol, surface = key.split("_")
        name = {"C":"center", "Y":"near_y_edge", "X":"near_x_edge"}[symbol]
        record["disks"][surface][name]["mean_von_mises"] += delta


class MetricContracts(unittest.TestCase):
    def test_seven_distinct_physical_metrics_preserve_top_bottom_and_circle_mapping(self):
        reference = assessment_reference(); record = copy.deepcopy(reference)
        deltas = dict(zip(ERROR_KEYS, (12.,1.,3.,7.,13.,29.,61.)))
        for key, delta in deltas.items():
            change_metric(record,key,delta)
        record["diagnostic_max_VM"] = {"top": 1e20, "bottom": 1e30}
        result = assess(record, reference)
        self.assertEqual(tuple(result["values"]), ERROR_KEYS)
        self.assertEqual(set(result["relative_errors"]), set(ERROR_KEYS))
        self.assertEqual(set(result["primary_checks"]), set(ERROR_KEYS))
        self.assertEqual(set(result["strict_checks"]), set(ERROR_KEYS))
        self.assertEqual(len(result["relative_errors"]), 7)
        for key, delta in deltas.items():
            self.assertEqual(result["values"][key]-result["reference_values"][key], delta)
            self.assertAlmostEqual(result["relative_errors"][key], delta/result["reference_values"][key])
        self.assertNotIn("diagnostic_max_VM", result["relative_errors"])

    def test_primary_boundary_passes_and_strict_goal_separately_fails(self):
        reference = assessment_reference(); record = copy.deepcopy(reference)
        change_metric(record,"w",5.)
        for key in ERROR_KEYS[1:]:
            target=assess(reference,reference)["values"][key]
            change_metric(record,key,.05*target)
        result=assess(record,reference)
        self.assertTrue(result["primary_goal_met"])
        self.assertTrue(all(result["primary_checks"].values()))
        self.assertFalse(result["strict_goal_met"])
        self.assertFalse(any(result["strict_checks"].values()))

    def test_strict_boundary_passes_and_implies_primary(self):
        reference = assessment_reference(); record = copy.deepcopy(reference)
        change_metric(record,"w",1.)
        for key in ERROR_KEYS[1:]:
            target=assess(reference,reference)["values"][key]
            change_metric(record,key,.01*target)
        result=assess(record,reference)
        self.assertTrue(result["primary_goal_met"])
        self.assertTrue(result["strict_goal_met"])

    def test_one_bottom_disk_cannot_be_hidden_by_five_other_disks(self):
        reference = assessment_reference(); record = copy.deepcopy(reference)
        change_metric(record,"X_bottom",161.)
        result=assess(record,reference)
        self.assertFalse(result["primary_goal_met"])
        self.assertFalse(result["strict_goal_met"])
        self.assertEqual([k for k,v in result["primary_checks"].items() if not v],["X_bottom"])
        self.assertEqual([k for k,v in result["strict_checks"].items() if not v],["X_bottom"])

    def test_error_is_relative_to_reference_and_uses_absolute_difference(self):
        reference = assessment_reference(); record = copy.deepcopy(reference)
        change_metric(record,"w",-100.)
        result=assess(record,reference)
        self.assertEqual(result["relative_errors"]["w"],.2)
        self.assertNotEqual(result["relative_errors"]["w"],.25)


class PhysicalPointAndBudget(unittest.TestCase):
    def test_central_uz_is_bilinear_at_500_300_on_nonuniform_grid(self):
        x=np.array([0.,175.,470.,730.,1000.]);y=np.array([0.,110.,355.,600.])
        xx,yy=np.meshgrid(x,y)
        field=lambda a,b: .05+3e-4*a+7e-4*b+1e-6*a*b
        uz=field(xx,yy)
        scalar_tensor=np.zeros((len(y),len(x),6));scalar_tensor[:,:,0]=uz
        point=np.array([[CONFIG["a"]/2,CONFIG["b"]/2]])
        central=bilinear_stress(x,y,scalar_tensor,point)[0,0]
        self.assertNotIn(500.,x);self.assertNotIn(300.,y)
        self.assertAlmostEqual(central,field(500.,300.),places=14)
        self.assertNotAlmostEqual(central,uz[np.argmin(abs(y-300.)),np.argmin(abs(x-500.))],places=8)

    def test_over_budget_proposal_rejected_before_touching_native_session(self):
        class UntouchedSession:
            def __init__(self):self.calls=[]
            def __getattr__(self,name):
                self.calls.append(name)
                raise AssertionError("Budget rejection touched native session: "+name)
        session=UntouchedSession()
        with tempfile.TemporaryDirectory(prefix="strip-contract-") as temporary:
            runtime=Path(temporary)/"runtime";runtime.mkdir()
            output=Path(temporary)/"public";output.mkdir()
            with self.assertRaisesRegex(ValueError,"cap before solve"):
                execute_case(session,"over_budget",np.linspace(0,1000,81),np.linspace(0,600,25),
                             CONFIG,{"active_dof_cap":5382},runtime,output)
            self.assertEqual(session.calls,[])
            self.assertFalse((output/"over_budget"/"measurement.json").exists())
            self.assertFalse((runtime/"over_budget"/"solve_completed.json").exists())


class ReferenceFreeSelector(unittest.TestCase):
    def test_graph_schema_and_normalization_do_not_copy_reference_or_goals(self):
        class ForbiddenReference:
            def __float__(self):raise AssertionError("Reference entered selector input")
        poison=ForbiddenReference()
        mesh=build_mesh([0.,500.,1000.],[0.,300.,600.],CONFIG)
        mesh["reference"]=poison
        resultants=np.arange(32,dtype=float).reshape(4,8)
        indicator={"element_ids":mesh["element_ids"],"eta2":[1.,2.,3.,4.],
                   "parts_per_element":{"membrane":[1.,2.,3.,4.],"bending":[0.]*4,"shear":[0.]*4},
                   "reference_values":poison,"relative_errors":poison}
        physics={"a_mm":1000.,"b_mm":600.,"pressure_N_per_mm2":.005,
                 "young_N_per_mm2":210000.,"poisson":.3,"thickness_mm":8.,
                 "reference":poison,"primary_goal":poison,"strict_goal":poison}
        observation=build_observation(mesh,resultants,indicator,physics)
        expected={"schema_version","node_semantics","element_ids","eta2","feature_names","node_features",
                  "edge_index","edge_features","edge_feature_names","edge_definition","normalization",
                  "geometry","native_resultants_global","resultant_components","indicator_parts_N_mm",
                  "training_or_inference_executed"}
        self.assertEqual(set(observation),expected)
        self.assertFalse(observation["training_or_inference_executed"])
        self.assertEqual(observation["normalization"]["N_and_Q_N_per_mm"],3.)
        self.assertEqual(observation["normalization"]["M_N"],1800.)
        self.assertFalse(any("reference" in name.lower() or "goal" in name.lower()
                             for name in observation["feature_names"]))
        np.testing.assert_array_equal(classical_selector(observation),[1.,2.,3.,4.])

    def test_classical_selector_reads_only_aligned_ids_and_eta_squared(self):
        class RestrictedObservation(dict):
            def __getitem__(self,key):
                if key not in ("element_ids","eta2"):
                    raise AssertionError("Selector requested forbidden feature: "+key)
                return super().__getitem__(key)
        observation=RestrictedObservation(element_ids=[11,17,23,31],eta2=[1.,4.,2.,3.],
                                          reference_values=object(),relative_errors=object(),goals=object())
        scores=classical_selector(observation)
        marked,_=dorfler_mark(observation["element_ids"],scores)
        self.assertEqual(marked,[17,31])


if __name__ == "__main__":
    unittest.main()
