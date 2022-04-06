from pyscf import gto, dft
from vdw import to_mbd


def obtain_ts_vdw_from_mf():
    mol = gto.Mole(basis="cc-pVDZ",
    atom="""
        O  0.  0.  0.
        H  0.  0.  1.
        H  0.  1.  0.
        O  0.  0.  2.
        H  0.  0.  3.
        H  0.  1.  2.
    """).build()

    # Example:
    mf_bare = dft.RKS(mol, xc="PBE").run()
    mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="ts").run()
    print(mf.e_tot - mf_bare.e_tot)
    # FHI-Aims: -0.00021265 Ha
    mf_bare_grad = mf_bare.nuc_grad_method().run()
    mf_grad = mf.nuc_grad_method().run()
    print(mf_grad.de - mf_bare_grad.de)
    # FHI-Aims:
    # 0.00000000    -0.00002001     0.00001925
    # 0.00000000    -0.00000206    -0.00002847
    # 0.00000000     0.00001193    -0.00004670
    # 0.00000000    -0.00002185     0.00005286
    # 0.00000000     0.00000911    -0.00007099
    # 0.00000000     0.00002288     0.00007404

    # set parameters for unknown xc
    # WARNING!!! Following code does not suggests cam-b3lyp use sr=1.05! Only for API illustration!
    #            Parameter sr is s_R, where for PBE this value is 0.94
    mf = to_mbd(dft.RKS(mol, xc="CAM-B3LYP"), variant="ts", sr=1.05).run()
    print(mf.e_vdw)


if __name__ == '__main__':
    print()
    print("=== obtain_ts_vdw_from_mf ===")
    obtain_ts_vdw_from_mf()
