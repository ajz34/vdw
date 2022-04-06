from dftd3.parameters import get_damping_param
from dftd3.interface import (
    DispersionModel, RationalDampingParam, ZeroDampingParam,
    ModifiedRationalDampingParam, ModifiedZeroDampingParam, OptimizedPowerDampingParam)
from vdw.wrapper import wrapper

from pyscf import gto, scf, mcscf, lib
from pyscf.lib import logger
import numpy as np


class WithDFTD3(lib.StreamObject):
    mol = None
    model = None
    xc = None
    param = None
    version = None
    atm = None
    verbose = None
    do_grad = False

    _param = None
    _param_output = None
    _version = None
    _result = None

    def __init__(self, mf, version="bj", xc=None, param=None, atm=False):
        self.parse_mf(mf)
        self.version = version
        if xc:
            self.xc = xc
        self.param = param
        self.atm = atm
        self.verbose = mf.verbose

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
            self._version = "bj"
        elif self.version == 3 or self.version.lower() == "zero":
            param_cls = ZeroDampingParam
            self._version = "zero"
        elif self.version == 6 or self.version.lower() == "bjm":
            param_cls = ModifiedRationalDampingParam
            self._version = "bjm"
        elif self.version == 5 or self.version.lower() == "zerom":
            param_cls = ModifiedZeroDampingParam
            self._version = "zerom"
        elif self.version.lower() == "op":
            param_cls = OptimizedPowerDampingParam
            self._version = "op"
        else:
            raise ValueError("DFTD3 version is not recognized!")

        if self.param:
            param = self.param.copy()
            if "s9" not in param and not self.atm:
                param["s9"] = 0.
            self._param = param_cls(**param)
            self._param_output = param
        else:
            self._param = param_cls(method=self.xc, atm=self.atm)
            self._param_output = get_damping_param(method=self.xc, defaults=[self._version])

    def dump_flags(self, verbose=0):
        self.parse_config()
        logger.info(self, "[INFO] DFTD3 Parameter")
        for k in self._param_output:
            logger.info(self, "       {:6} = {:8.4f}".format(k, self._param_output[k]))
        logger.info(self, "       version: " + self._version)

    @property
    def eng(self):
        self.parse_config()
        model = DispersionModel(self.mol.atom_charges(), self.mol.atom_coords())
        self._result = model.get_dispersion(self._param, grad=self.do_grad)
        return self._result["energy"]

    @property
    def grad(self):
        if self._result is not None and "gradient" in self._result:
            return self._result["gradient"]
        self.parse_config()
        model = DispersionModel(self.mol.atom_charges(), self.mol.atom_coords())
        self._result = model.get_dispersion(self._param, grad=True)
        return self._result["gradient"]


def to_dftd3(mf, do_grad=False, **kwargs):
    wrap = wrapper(WithDFTD3, mf, **kwargs)
    wrap.with_vdw.do_grad = do_grad
    return wrap


def main():
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
    """, basis="6-31G", unit="Angstrom", verbose=4).build()

    print()
    mf = dft.RKS(mol, xc="PBE0")
    # BJ PBE0
    WithDFTD3(mf).dump_flags(0)
    assert np.allclose(WithDFTD3(mf).eng, -2.9109185161490E-02)
    assert np.allclose(WithDFTD3(mol, xc="PBE0").eng, -2.9109185161490E-02)
    assert np.allclose(WithDFTD3(mol, version="bj", param={"s8": 1.2177, "a1": 0.4145, "a2": 4.8593}).eng,
                       -2.9109185161490E-02)
    # Zero PBE0
    assert np.allclose(WithDFTD3(mf, version="zero").eng, -1.4174415944614E-02)

    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = 'cc-pvdz'
    mol.build()
    mf = to_dftd3(scf.RHF(mol))
    e0 = mf.kernel()
    print(e0 - -75.99396273778923)

    mfs = mf.as_scanner()
    e2 = mfs(''' O                 -0.00000000    0.00000000   -0.11181188
                 H                 -0.00000000   -0.84695236    0.59109389
                 H                 -0.00000000    0.89830571    0.52404783 ''')
    e1 = mfs(''' O                  0.00000000    0.00000000   -0.10981188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 ''')
    g = mf.nuc_grad_method().kernel()
    print((e1 - e2)/0.002 * lib.param.BOHR - g[0, 2])


if __name__ == '__main__':
    main()
