from pymbd.fortran import MBDGeom
from pymbd.pymbd import vdw_params
from vdw.util.hirshfeld import HirshfeldAnalysis
from pyscf import scf, mcscf, lib
import numpy as np

from vdw.wrapper import wrapper

DEFAULT_TS_SR = {
    "pbe": 0.94,
    "pbe0": 0.96,
    "hse": 0.96,
    "blyp": 0.62,
    "b3lyp": 0.84,
    "revpbe": 0.60,
    "am05": 0.84,
}

DEFAULT_RSSCS_BETA = {
    "pbe": 0.83,
    "pbe0": 0.85,
    "hse": 0.85,
}

DEFAULT_NL_BETA = {
    "pbe": 0.81,
    "pbe0": 0.83,
    "hse": 0.83,
}

DEFAULT_TS_BETA = {
    "pbe": 0.81,
    "pbe0": 0.83,
    "hse": 0.83,
}

DEFAULT_SCS_A = {
    "pbe": 2.56,
    "pbe0": 2.53,
    "hse": 2.53,
}


class WithMBD(lib.StreamObject):
    mol = None
    mf = None
    xc = None
    hirshfeld_result = None

    a = None
    beta = None
    sr = None
    variant = None
    damping = None
    verbose = None
    do_grad = False

    _result = {}

    def __init__(self, mf, variant, beta=None, a=None, sr=None, damping=None, do_grad=False):
        self.parse_mf(mf)
        self.variant = variant
        self.beta = beta
        self.a = a
        self.sr = sr
        self.damping = damping
        self.do_grad = do_grad

    def parse_mf(self, mf):
        assert(isinstance(mf, scf.hf.SCF) or
               isinstance(mf, mcscf.casci.CASCI))
        self.mol = mf.mol
        self.mf = mf
        if isinstance(mf, mcscf.casci.CASCI):
            self.xc = 'hf'
        else:
            self.xc = getattr(mf, 'xc', 'HF').replace(' ', '')
        return self

    def parse_config(self):
        xc = self.xc.lower().replace(" ", "").replace("-", "")
        variant = self.variant.lower().replace(" ", "").replace("-", "").replace("@", "")
        if variant in ("ts", "tsvdw"):
            self.sr = DEFAULT_TS_SR.get(xc)
            if self.sr is None:
                raise ValueError("XC for TS-vDW not recognized!")
        elif variant in ("mbdrsscs", "rsscs"):
            self.a = 6.0
            self.beta = DEFAULT_RSSCS_BETA.get(xc)
            self.damping = "fermi,dip"
            if self.beta is None:
                raise ValueError("XC for MBD@RSSCS not recognized!")
        elif variant in ("mbdnl", "nl"):
            self.a = 6.0
            self.beta = DEFAULT_NL_BETA.get(xc)
            self.damping = "fermi,dip"
            if self.beta is None:
                raise ValueError("XC for MBD@NL not recognized!")
        elif variant in ("mbdts", ):
            self.a = 6.0
            self.beta = DEFAULT_TS_BETA.get(xc)
            self.damping = "fermi"
            if self.beta is None:
                raise ValueError("XC for MBD@TS not recognized!")
        elif variant in ("mbdscs", "scs"):
            self.beta = 0.0
            self.a = DEFAULT_SCS_A.get(xc)
            self.damping = "fermi,dip"
            if self.a is None:
                raise ValueError("XC for MBD@SCS not recognized!")
        else:
            raise ValueError("Could not parse method here.")
        return self

    def parse_hirshfeld(self):
        self.hirshfeld_result = HirshfeldAnalysis(self.mf).run().result
        return self

    def perform_calc(self):
        if not (self.a or self.beta or self.sr):
            self.parse_config()
        if self.hirshfeld_result is None:
            self.parse_hirshfeld()

        mol = self.mol
        species = [mol.atom_symbol(i) for i in range(mol.natm)]
        alpha_free, C6_free, R0_free = (
            np.array([vdw_params[sp][param] for sp in species])
            for param in 'alpha_0(TS) C6(TS) R_vdw(TS)'.split()
        )
        V_free, V_eff = [self.hirshfeld_result[token] for token in ("V_free", "V_eff")]
        V_ratio = V_eff / V_free
        alpha_eff, C6_eff, R0_eff = alpha_free * V_ratio, C6_free * V_ratio**2, R0_free * V_ratio**(1/3)
        mbd_obj = MBDGeom(self.mol.atom_coords())
        variant = self.variant.lower().replace(" ", "").replace("-", "").replace("@", "")
        if variant in ("ts", "tsvdw"):
            result = mbd_obj.ts_energy(alpha_eff, C6_eff, R0_eff, self.sr, force=self.do_grad)
        else:
            if variant in ("mbdrsscs", "rsscs"):
                var = "rsscs"
            elif variant in ("mbdscs", "scs"):
                var = "scs"
            elif variant in ("mbd", "mbdnl", "nl", "mbdts", "plain"):
                var = "plain"
            else:
                raise ValueError("Unknown variant!")
            result = mbd_obj.mbd_energy(alpha_eff, C6_eff, R0_eff, beta=self.beta, a=self.a,
                                        variant=var, damping=self.damping, force=self.do_grad)
        if self.do_grad:
            self._result["energy"], self._result["gradient"] = result
        else:
            self._result["energy"] = result
        return self

    @property
    def eng(self):
        self.perform_calc()
        return self._result["energy"]

    @property
    def grad(self):
        if "gradient" in self._result:
            return self._result["gradient"]
        if self.do_grad is False:
            self.do_grad = True
        self.perform_calc()
        return self._result["gradient"]

    def dump_flags(self, verbose=0):
        pass


def to_mbd(mf, **kwargs):
    wrap_cls = wrapper(WithMBD, mf, return_instance=False, **kwargs)
    mf_cls = mf.__class__

    class MBDInner(wrap_cls):

        def energy_nuc(self):
            return mf.energy_nuc()

        def kernel(self, *args, **kwargs):
            mf_cls.kernel(self, *args, **kwargs)
            self.e_vdw = self.with_vdw.eng
            self.e_tot += self.e_vdw
            return self.e_tot

    with_vdw = WithMBD(mf, **kwargs)
    obj = MBDInner(mf, with_vdw)
    obj.with_vdw.mf = obj
    return obj


def main():
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 3; H 0 0.5 3.5; H 0 -0.5 3.5
    """
    mol.basis = "cc-pVDZ"
    mol.build()
    mf = dft.RKS(mol, xc="PBE")
    mf_mbd = to_mbd(mf, variant="rsscs").run()
    print(mf_mbd.e_tot)
    print(mf_mbd.e_vdw)


if __name__ == '__main__':
    main()
