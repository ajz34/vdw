from pyscf import gto, dft
from vdw import WithDFTD4, to_dftd4


def obtain_dftd4_from_molecule():
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
    # dftd4 --func pbe diben.xyz --grad

    # use pbe xc
    with_vdw = WithDFTD4(mol, xc="PBE")
    print(with_vdw.eng)
    # expect value: -3.7373975545997E-02 Eh
    print(with_vdw.grad)
    # expect value:
    #     1    6 C    -6.963E-05 -3.798E-19  4.082E-03
    #    24    1 H     4.270E-05 -7.396E-05  1.109E-04

    # use self-defined parameters
    with_vdw = WithDFTD4(mol, param={"s8": 0.9595, "a1": 0.3857, "a2": 4.8069})
    print(with_vdw.eng)


def obtain_dftd4_from_mf():
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
    mf = to_dftd4(dft.RKS(mol, xc="PBE")).run()
    # dftd4 --func pbe diwater.xyz --grad
    print(mf.e_tot - mf_bare.e_tot)
    # expect value: -1.2272300355510E-03 Eh
    mf_bare_grad = mf_bare.nuc_grad_method().run()
    mf_grad = mf.nuc_grad_method().run()
    print(mf_grad.de - mf_bare_grad.de)
    # expect value:
    #     1    8 O     0.000E+00  2.918E-05 -5.534E-05
    #     6    1 H     0.000E+00 -2.760E-05  1.008E-05


if __name__ == '__main__':
    print()
    print("=== obtain_dftd4_from_molecule ===")
    obtain_dftd4_from_molecule()
    print("=== obtain_dftd4_from_mf ===")
    obtain_dftd4_from_mf()
