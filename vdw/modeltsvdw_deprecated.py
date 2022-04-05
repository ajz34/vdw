from vdw.util.sph_dft_atom_ks import get_atm_nrks, free_atom_info
from vdw.util.vdw_param import vdw_param
from vdw.wrapper import wrapper
from pyscf import gto, dft, lib, scf, mcscf
from pyscf.lib import logger
import warnings
import numpy as np

DICT_sR = {
    "pbe": 0.94,
    "pbe0": 0.96,
    "blyp": 0.62,
    "b3lyp": 0.84,
    "revpbe": 0.60,
    "m06l": 1.26,
    "m06": 1.16,
}


class WithTSvDW(lib.StreamObject):
    mol = None
    mf = None
    xc = None
    verbose = None
    sR = None

    _result = {}
    _d = 20

    def __init__(self, mf, xc=None, sR=None):
        self.parse_mf(mf)
        self.verbose = mf.verbose
        if xc:
            self.xc = xc
        self.sR = sR

    def parse_mf(self, mf):
        assert(isinstance(mf, scf.hf.SCF) or
               isinstance(mf, mcscf.casci.CASCI))
        self.mf = mf
        self.mol = mf.mol
        if isinstance(mf, mcscf.casci.CASCI):
            self.xc = 'hf'
        else:
            self.xc = getattr(mf, 'xc', 'HF').replace(' ', '')

    def parse_config(self):
        if self.sR:
            return
        xc = self.xc.lower().replace(" ", "").replace("-", "")
        self.sR = DICT_sR[xc]

    def perform_params(self):
        result = self._result
        mf = self.mf
        mol = self.mol  # type: gto.Mole
        ni = dft.numint.NumInt()
        grids = getattr(mf, "grids", None)
        if grids is None:
            grids = dft.Grids(mol)
            grids.atom_grid = (77, 302)
            grids.build()
        rho = ni.get_rho(mol, mf.make_rdm1(), grids)

        result["mf_elem"] = {}
        result["V_free"] = {}
        result["spl_free"] = {}
        mf_elems = get_atm_nrks(mf)
        for elem in mf_elems:
            mf_elem = mf_elems[elem]
            result["mf_elem"][elem] = mf_elem
            result["V_free"][elem], result["spl_free"][elem] = free_atom_info(mf_elem)

        coords_atoms = grids.coords[None, :, :] - mol.atom_coords()[:, None, :]
        rad_atoms = np.linalg.norm(coords_atoms, axis=-1)
        rho_free = np.empty((mol.natm, len(grids.coords)))
        V_free = np.empty(mol.natm)
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            rho_free[atom] = result["spl_free"][elem](rad_atoms[atom])
            V_free[atom] = result["V_free"][elem]
        weights_free = rho_free / (rho_free.sum(axis=0) + 1e-15)
        V_eff = (weights_free * rho * rad_atoms**3 * grids.weights).sum(axis=-1)
        V_ratio = V_eff / V_free

        alpha_free = np.zeros(mol.natm)
        C6_free = np.zeros(mol.natm)
        R0_free = np.zeros(mol.natm)
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            alpha_free[atom], C6_free[atom], R0_free[atom] = vdw_param[elem]

        alpha_eff = alpha_free * V_ratio
        C6_eff = C6_free * V_ratio**2
        R0_eff = R0_free * V_ratio**(1/3)

        result["V_free"] = V_free
        result["alpha_free"] = alpha_free
        result["C6_free"] = C6_free
        result["R0_free"] = R0_free
        result["V_eff"] = V_eff
        result["alpha_eff"] = alpha_eff
        result["C6_eff"] = C6_eff
        result["R0_eff"] = R0_eff

    def perform_eng(self):
        mol = self.mol
        result = self._result
        if "alpha_eff" not in result:
            self.perform_params()

        alpha_eff = result["alpha_eff"]
        C6_eff = result["C6_eff"]
        R0_eff = result["R0_eff"]

        dist_comp = mol.atom_coords()[:, None, :] - mol.atom_coords()[None, :, :]
        dist = np.linalg.norm(dist_comp, axis=-1)
        np.fill_diagonal(dist, np.inf)
        C6_pair = 2 * C6_eff[:, None] * C6_eff[None, :] / (
            + alpha_eff[None, :] / alpha_eff[:, None] * C6_eff[:, None]
            + alpha_eff[:, None] / alpha_eff[None, :] * C6_eff[None, :])
        R0_pair = R0_eff[:, None] + R0_eff[None, :]

        d = self._d
        f_damp_pair = 1 / (1 + np.exp(- d * (dist / self.sR / R0_pair - 1)))
        vdw_pair = -0.5 * C6_pair * dist**(-6) * f_damp_pair
        eng_vdw = vdw_pair.sum()

        result["vdw_pair"] = vdw_pair
        result["C6_pair"] = C6_pair
        result["eng_vdw"] = eng_vdw

    def perform_grad(self):
        warnings.warn("This gradient implementation for TS-vDW does not count density relaxization into account.\n"
                      "Use this gradient with caution!")
        mol = self.mol
        result = self._result
        if "eng_vdw" not in result:
            self.perform_eng()

        R0_eff = result["R0_eff"]
        C6_pair = result["C6_pair"]

        dist_comp = mol.atom_coords()[:, None, :] - mol.atom_coords()[None, :, :]
        dist = np.linalg.norm(dist_comp, axis=-1)
        np.fill_diagonal(dist, np.inf)
        R0_pair = R0_eff[:, None] + R0_eff[None, :]

        d = self._d
        g_damp_pair = np.exp(- d * (dist / self.sR / R0_pair - 1))
        f_damp_pair = 1 / (1 + g_damp_pair)
        f_damp_pair_grad = d / self.sR / R0_pair * g_damp_pair / (1 + g_damp_pair)**2
        grad_contrib_1 = 6 * ((C6_pair / dist**8 * f_damp_pair)[:, :, None] * dist_comp).sum(axis=-2)
        grad_contrib_2 = - ((C6_pair / dist**7 * f_damp_pair_grad)[:, :, None] * dist_comp).sum(axis=-2)
        result["grad_vdw"] = grad_contrib_1 + grad_contrib_2

    def dump_flags(self, verbose=0):
        self.parse_config()
        logger.info(self, "[INFO] TS-vDW Output")
        logger.info(self, str(self._result))

    @property
    def eng(self):
        self.parse_config()
        self.perform_eng()
        return self._result["eng_vdw"]

    @property
    def grad(self):
        self.parse_config()
        self.perform_grad()
        return self._result["grad_vdw"]


def to_tsvdw(mf, **kwargs):
    wrap_cls = wrapper(WithTSvDW, mf, return_instance=False, **kwargs)
    mf_cls = mf.__class__

    class TSvDWInner(wrap_cls):

        def energy_nuc(self):
            return mf.energy_nuc()

        def kernel(self, *args, **kwargs):
            result = mf_cls.kernel(self, *args, **kwargs)
            self.e_vdw = self.with_vdw.eng
            self.e_tot += self.e_vdw
            return self.e_tot

    with_vdw = WithTSvDW(mf, **kwargs)
    obj = TSvDWInner(mf, with_vdw)
    obj.with_vdw.mf = obj
    return obj


def main():
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 3; H 0 0.5 3.5; H 0 -0.5 3.5
    """
    mol.basis = "cc-pVDZ"
    mol.build()

    mf = to_tsvdw(dft.RKS(mol, xc="PBE")).run()
    print(mf.e_tot)
    print(mf.e_vdw)

    mf = dft.RKS(mol, xc="PBE").run()
    w0 = WithTSvDW(mf)
    print(mf.e_tot)
    print(w0.eng)
    print(w0.grad)


def mbd():
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 3; H 0 0.5 3.5; H 0 -0.5 3.5
    """
    mol.basis = "cc-pVDZ"
    mol.build()
    mf = dft.RKS(mol, xc="PBE").run()
    wv = WithTSvDW(mf)
    wv.parse_config()
    wv.perform_params()
    print(wv.eng)

    from pymbd.fortran import MBDGeom
    wm = MBDGeom(mol.atom_coords())
    print(wm.ts_energy(wv._result["alpha_eff"], wv._result["C6_eff"], wv._result["R0_eff"], 0.94, force=True))


if __name__ == '__main__':
    main()
    mbd()
