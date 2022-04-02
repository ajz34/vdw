from pyscf import gto, scf, mcscf
from dftd4.interface import DispersionModel, DampingParam
import numpy as np


class WithDFTD4:
    mol = None
    model = None
    xc = None
    param = None

    _param = None

    def __init__(self, mf, xc=None, param=None):
        self.parse_mf(mf)
        if xc:
            self.xc = xc
        self.param = param

    def parse_mf(self, mf):
        if isinstance(mf, gto.Mole):
            self.mol = mf
            return
        assert(isinstance(mf, scf.hf.SCF) or
               isinstance(mf, mcscf.casci.CASCI))
        self.mol = mf.mol
        if isinstance(mf, mcscf.casci.CASCI):
            self.xc = 'hf'
        else:
            self.xc = getattr(mf, 'xc', 'HF').replace(' ', '')

    def parse_config(self):
        if self.param:
            self._param = DampingParam(**self.param)
        else:
            self._param = DampingParam(method=self.xc)

    @property
    def eng(self):
        self.parse_config()
        model = DispersionModel(self.mol.atom_charges(), self.mol.atom_coords())
        return model.get_dispersion(self._param, grad=False)["energy"]

    @property
    def grad(self):
        self.parse_config()
        model = DispersionModel(self.mol.atom_charges(), self.mol.atom_coords())
        return model.get_dispersion(self._param, grad=True)["gradient"]


if __name__ == '__main__':
    from pyscf import dft
    mol = gto.Mole(atom="""
    C   1.40000000   0.00000000   0.00000000
    C   0.70000000   1.21243557   0.00000000
    C  -0.70000000   1.21243557   0.00000000
    C  -1.40000000   0.00000000   0.00000000
    C  -0.70000000  -1.21243557   0.00000000
    C   0.70000000  -1.21243557   0.00000000
    H   2.49000000   0.00000000   0.00000000
    H   1.24500000   2.15640326   0.00000000
    H  -1.24500000   2.15640326   0.00000000
    H  -2.49000000   0.00000000   0.00000000
    H  -1.24500000  -2.15640326   0.00000000
    H   1.24500000  -2.15640326   0.00000000
    C   1.40000000   0.00000000   2.00000000
    C   0.70000000   1.21243557   2.00000000
    C  -0.70000000   1.21243557   2.00000000
    C  -1.40000000   0.00000000   2.00000000
    C  -0.70000000  -1.21243557   2.00000000
    C   0.70000000  -1.21243557   2.00000000
    H   2.49000000   0.00000000   2.00000000
    H   1.24500000   2.15640326   2.00000000
    H  -1.24500000   2.15640326   2.00000000
    H  -2.49000000   0.00000000   2.00000000
    H  -1.24500000  -2.15640326   2.00000000
    H   1.24500000  -2.15640326   2.00000000
    """, basis="6-31G", unit="Angstrom").build()

    print()
    mf = dft.RKS(mol, xc="PBE0")
    assert np.allclose(WithDFTD4(mf).eng, -3.1667289823883E-02)