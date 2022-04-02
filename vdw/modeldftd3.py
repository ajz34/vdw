from pyscf import gto, scf, mcscf
from dftd3.interface import (
    DispersionModel, RationalDampingParam, ZeroDampingParam,
    ModifiedRationalDampingParam, ModifiedZeroDampingParam, OptimizedPowerDampingParam)
import numpy as np


class WithDFTD3:
    mol = None
    model = None
    xc = None
    param = None
    version = None
    atm = None

    _param = None

    def __init__(self, mf, version="bj", xc=None, param=None, atm=False):
        self.parse_mf(mf)
        self.version = version
        if xc:
            self.xc = xc
        self.param = param
        self.atm = atm

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
        if self.version == 4 or self.version.lower() == "bj":
            param_cls = RationalDampingParam
        elif self.version == 3 or self.version.lower() == "zero":
            param_cls = ZeroDampingParam
        elif self.version == 6 or self.version.lower() == "bjm":
            param_cls = ModifiedRationalDampingParam
        elif self.version == 5 or self.version.lower() == "zerom":
            param_cls = ModifiedZeroDampingParam
        elif self.version.lower() == "op":
            param_cls = OptimizedPowerDampingParam
        else:
            raise ValueError("DFTD3 version is not recognized!")

        if self.param:
            param = self.param.copy()
            if "s9" not in param and not self.atm:
                param["s9"] = 0.
            self._param = param_cls(**param)
        else:
            self._param = param_cls(method=self.xc, atm=self.atm)

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
    # BJ PBE0
    assert np.allclose(WithDFTD3(mf).eng, -2.9109185161490E-02)
    assert np.allclose(WithDFTD3(mol, xc="PBE0").eng, -2.9109185161490E-02)
    assert np.allclose(WithDFTD3(mol, version="bj", param={"s8": 1.2177, "a1": 0.4145, "a2": 4.8593}).eng, -2.9109185161490E-02)
    # Zero PBE0
    assert np.allclose(WithDFTD3(mf, version="zero").eng, -1.4174415944614E-02)
