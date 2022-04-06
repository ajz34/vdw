from pyscf import gto, dft
from vdw import to_mbd


def obtain_mbd_rsscs_from_mf():
    mol = gto.Mole(basis="cc-pVDZ",
    atom="""
        O  0.  0.  0.
        H  0.  0.  1.
        H  0.  1.  0.
        O  0.  0.  2.
        H  0.  0.  3.
        H  0.  1.  2.
    """).build()
    mf_bare = dft.RKS(mol, xc="PBE").run()
    mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="rsscs").run()
    print(mf.e_tot - mf_bare.e_tot)
    # FHI-Aims: -0.00124717 Ha
    mf_bare_grad = mf_bare.nuc_grad_method().run()
    mf_grad = mf.nuc_grad_method().run()
    print(mf_grad.de - mf_bare_grad.de)


if __name__ == '__main__':
    print()
    print("=== obtain_mbd_rsscs_from_mf ===")
    obtain_mbd_rsscs_from_mf()
