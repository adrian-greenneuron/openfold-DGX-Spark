"""Guard against accepting incomplete outputs or misleading structural agreement."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark import validate_pdb
from compare_structures import compare

def pdb(xyz,resname='ALA'):
    return ''.join(f'ATOM  {i:5d}  CA  {resname} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{80.:6.2f}           C\n'
                   for i,(x,y,z) in enumerate(xyz,1))

class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.a=Path(self.tmp.name)/'a.pdb';self.b=Path(self.tmp.name)/'b.pdb'
        self.xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.]])
        self.a.write_text(pdb(self.xyz));self.b.write_text(pdb(self.xyz))
    def test_valid_complete_prediction(self):self.assertEqual(validate_pdb(self.a,4)['residues'],4)
    def test_incomplete_prediction_is_rejected(self):
        with self.assertRaises(ValueError):validate_pdb(self.a,5)
    def test_nonfinite_prediction_is_rejected(self):
        self.xyz[0,0]=float('nan');self.a.write_text(pdb(self.xyz))
        with self.assertRaises(ValueError):validate_pdb(self.a,4)
    def test_missing_prediction_is_rejected(self):
        with self.assertRaises(FileNotFoundError):validate_pdb(Path(self.tmp.name)/'missing.pdb',4)
    def test_rigid_motion_keeps_local_agreement(self):
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        self.b.write_text(pdb(self.xyz@r+4))
        result=compare(self.a,self.b)
        self.assertAlmostEqual(result['ca_lddt'],1.)
        self.assertLess(result['ca_rmsd_angstrom'],1e-10)
    def test_changed_structure_is_detected(self):
        self.xyz[0,0]=30;self.b.write_text(pdb(self.xyz))
        self.assertLess(compare(self.a,self.b)['ca_lddt'],1.)
    def test_residue_mismatch_is_rejected(self):
        self.b.write_text(pdb(self.xyz,'GLY'))
        with self.assertRaises(ValueError):compare(self.a,self.b)
    def test_empty_outputs_cannot_pass(self):
        self.a.write_text('');self.b.write_text('')
        with self.assertRaises(ValueError):compare(self.a,self.b)

if __name__=='__main__':unittest.main()
