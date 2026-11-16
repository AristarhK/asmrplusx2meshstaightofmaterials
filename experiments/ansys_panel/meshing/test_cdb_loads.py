"""Regression checks for native blocked loads; no MAPDL process is used."""
from pathlib import Path
import tempfile
import unittest

from cdb_loads import parse_cdb_loads


def dline(node=7, dof="UZ", real=0, imag=0):
    return f"{node:9d}{dof:6s}{real:16.9E}{imag:16.9E}"


def sfeline(elem=101, face=1, kval=1, values=(.005,) * 4):
    return f"{elem:9d}{face:4d}{kval:4d}" + "".join(f"{x:20.9E}" for x in values)


def sfeblock(real_lines, imag_lines, count=1):
    return "\n".join([f"SFEBLOCK,4,PRES,{count},{16 * count},0", "(i9,i4,i4,6(pg20.9))",
                       *real_lines, *imag_lines, "SFE,end,LOC,-1,"])


class NativeLoads(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.cdb"
            path.write_text(text, encoding="utf-8")
            return parse_cdb_loads(path)

    def test_fixed_width_dblock_preserves_noncontiguous_ids_and_complex_values(self):
        rows = self.parse("DBLOCK,2\n(i9,a6,2(pg16.9))\n" + dline() + "\n"
                          + dline(993, "ROTX", -1.25e-7, 2e-9) + "\n       -1\n")
        self.assertEqual(rows[0], ["D", "7", "UZ", "0", "0"])
        self.assertEqual(rows[1][:3], ["D", "993", "ROTX"])
        self.assertAlmostEqual(float(rows[1][3]), -1.25e-7)
        self.assertAlmostEqual(float(rows[1][4]), 2e-9)

    def test_real_and_zero_imag_pressure_become_one_four_corner_record(self):
        rows = self.parse(sfeblock([sfeline()], [sfeline(kval=2, values=(0,) * 4)]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:5], ["SFE", "101", "1", "PRES", "1"])
        self.assertEqual([float(x) for x in rows[0][5:]], [.005] * 4)

    def test_duplicates_remain_visible(self):
        rows = self.parse("DBLOCK,2\n(i9,a6,2(pg16.9))\n" + dline() + "\n"
                          + dline() + "\n-1\n" + sfeblock([sfeline()] * 2,
                            [sfeline(kval=2, values=(0,) * 4)] * 2, count=2))
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[2], rows[3])
        self.assertEqual(len(rows), 4)

    def test_nonzero_imaginary_pressure_is_flagged(self):
        rows = self.parse(sfeblock([sfeline()], [sfeline(kval=2, values=(0, 1e-8, 0, 0))]))
        self.assertEqual(rows[1][0], "SFE_IMAG")

    def test_unblocked_uniform_default_and_partial_blank_rule(self):
        rows = self.parse("D,7,UZ,0\nSFE,101,1,PRES,0,5D-3\nSFE,909,1,PRES,1,.005,,.006,\n")
        self.assertEqual(rows[0], ["D", "7", "UZ", "0", "0"])
        self.assertEqual([float(x) for x in rows[1][5:]], [.005] * 4)
        self.assertEqual([float(x) for x in rows[2][5:]], [.005, 0, .006, 0])

    def test_other_loads_and_constraints_are_not_silently_discarded(self):
        rows = self.parse("F,7,FZ,1\nIC,7,UZ,0\nCP,1,UX,7,8\nCE,1,0,7,UX,1\n"
                          "SF,7,PRES,.005\nBF,7,TEMP,30\nBFE,101,TEMP,1,30\n"
                          "BFUNIF,TEMP,30\nACEL,0,0,1\n")
        self.assertEqual([row[0] for row in rows], ["F", "IC", "CP", "CE", "SF", "BF", "BFE", "BFUNIF", "ACEL"])

    def test_native_zero_defaults_and_geometry_are_not_applied_loads(self):
        self.assertEqual(self.parse("ETBLOCK,1,1\n(2i9,19a9)\n1 181\n-1\n"
                                   "SECBLOCK,1\n8,1,0,5\nBFUNIF,TEMP,_TINY\n"
                                   "ACEL,0,0,0\nOMEGA,0,0,0,0\nFINISH\n"), [])

    def test_rejects_unknown_command_block_and_orphan_data(self):
        for text in ("MYSTERYBLOCK,1\n1 2 3\n", "FUNKNOWN,7,1\n", "1 2 3\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_rejects_truncated_or_miscounted_blocks(self):
        cases = ["DBLOCK,2\n(i9,a6,2(pg16.9))\n" + dline() + "\n-1\n",
                 "DBLOCK,1\n(i9,a6,2(pg16.9))\n" + dline() + "\n",
                 sfeblock([sfeline()], [], count=1),
                 sfeblock([sfeline()], [sfeline(kval=2, values=(0,) * 4)], count=2),
                 sfeblock([sfeline()], [sfeline(kval=2, values=(0,) * 4)]).replace("SFE,end,LOC,-1,", "")]
        for text in cases:
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                self.parse(text)

    def test_rejects_unknown_formats_and_malformed_numeric_records(self):
        for text in ("DBLOCK,1\n(i8,a6,2(pg16.9))\n" + dline() + "\n-1\n",
                     "SFE,ALL,1,PRES,0,.005\n", "D,7,UZ,nan\n",
                     sfeblock([sfeline(kval=3)], [sfeline(kval=2, values=(0,) * 4)])):
            with self.subTest(text=text[:40]), self.assertRaises(ValueError):
                self.parse(text)


if __name__ == "__main__":
    unittest.main()
