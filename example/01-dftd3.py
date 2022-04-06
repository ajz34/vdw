from pyscf import gto, dft
from vdw import WithDFTD3, to_dftd3


def obtain_dftd3_from_molecule():
    mol = gto.Mole(basis="cc-pVDZ",
    atom="""
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
    """).build()
    # s-dftd3 --bj pbe diben.xyz --grad

    # use pbe xc
    with_vdw = WithDFTD3(mol, xc="PBE", version="bj")
    print(with_vdw.eng)
    # expect value: -3.3466717281444E-02 Eh
    print(with_vdw.grad)
    # expect value:
    #      1    6 C    -2.352E-05  4.321E-19  1.360E-03
    #     24    1 H     1.198E-04 -2.074E-04  1.445E-04

    # use self-defined parameters
    with_vdw = WithDFTD3(mol, version="bj", param={"s8": 0.7875, "a1": 0.4289, "a2": 4.4407})
    print(with_vdw.eng)


def obtain_dftd3_from_mf():
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
    mf = to_dftd3(dft.RKS(mol, xc="PBE"), version="zerom").run()
    # s-dftd3 --zerom pbe diwater.xyz --grad
    print(mf.e_tot - mf_bare.e_tot)
    # expect value: -2.0853937852139E-03 Eh
    mf_bare_grad = mf_bare.nuc_grad_method().run()
    mf_grad = mf.nuc_grad_method().run()
    print(mf_grad.de - mf_bare_grad.de)
    # expect value:
    #     1    8 O     0.000E+00 -2.759E-04 -4.003E-04
    #     6    1 H     0.000E+00  3.253E-04  1.180E-05


if __name__ == '__main__':
    print()
    print("=== obtain_dftd3_from_molecule ===")
    obtain_dftd3_from_molecule()
    print("=== obtain_dftd3_from_mf ===")
    obtain_dftd3_from_mf()
