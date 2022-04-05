from pyscf import scf, dft, mcscf
from vdw.util.sph_dft_atom_ks import get_atm_nrks, free_atom_info
import numpy as np


class HirshfeldAnalysis:
    mol = None
    mf = None
    xc = None

    result = {}

    def __init__(self, mf):
        self.parse_mf(mf)

    def parse_mf(self, mf):
        assert(isinstance(mf, scf.hf.SCF) or
               isinstance(mf, mcscf.casci.CASCI))
        self.mf = mf
        self.mol = mf.mol
        return self

    def perform_free_atom(self):
        result = self.result
        mf = self.mf
        mol = self.mol

        result["mf_elem"] = {}
        result["V_free_elem"] = {}
        result["spl_free_elem"] = {}
        mf_elems = get_atm_nrks(mf)
        for elem in mf_elems:
            mf_elem = mf_elems[elem]
            result["mf_elem"][elem] = mf_elem
            result["V_free_elem"][elem], result["spl_free_elem"][elem] = free_atom_info(mf_elem)

        result["V_free"] = np.zeros(mol.natm)
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            result["V_free"][atom] = result["V_free_elem"][elem]
        return self

    def perform_hirshfeld(self):
        result = self.result
        mf = self.mf
        mol = self.mol
        ni = dft.numint.NumInt()
        grids = getattr(mf, "grids", None)
        if grids is None:
            grids = dft.Grids(mol)
            grids.atom_grid = (77, 302)
            grids.build()
        rho = ni.get_rho(mol, mf.make_rdm1(), grids)

        coords_atoms = grids.coords[None, :, :] - mol.atom_coords()[:, None, :]
        rad_atoms = np.linalg.norm(coords_atoms, axis=-1)

        rho_free = np.empty((mol.natm, len(grids.coords)))
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            rho_free[atom] = result["spl_free_elem"][elem](rad_atoms[atom])
        weights_free = rho_free / (rho_free.sum(axis=0) + 1e-15)
        rho_eff = rho * weights_free
        V_eff = (rho_eff * rad_atoms ** 3 * grids.weights).sum(axis=-1)
        elec_eff = (rho_eff * grids.weights).sum(axis=-1)
        chrg_eff = - elec_eff + mol.atom_charges()
        dipole_eff = - (coords_atoms * rho_eff[:, :, None] * grids.weights[:, None]).sum(axis=-2)

        result["rho_free"] = rho_free
        result["weights_free"] = weights_free
        result["V_eff"] = V_eff
        result["charge_eff"] = chrg_eff
        result["dipole_eff"] = dipole_eff
        return self

    def run(self):
        self.perform_free_atom().perform_hirshfeld()
        return self


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 2; H 0 0 3; H 0 1 2
    """
    mol.basis = "cc-pVDZ"
    mol.build()
    mf = dft.RKS(mol, xc="PBE").run()

    anal = HirshfeldAnalysis(mf).run()
    print(anal.result["V_free"])
    print(anal.result["V_eff"])
