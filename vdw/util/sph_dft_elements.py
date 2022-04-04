# Non-relativistic spin-restricted spherically averaged exchange-only
# LDA a.k.a. Hartree-Fock-Slater configurations for use in atomic SAD
# calculations. Reference configurations from Phys. Rev. A 101, 012516
# (2020).
NRSRHFS_CONFIGURATION = [
    [0, 0, 0, 0],  # 0  GHOST
    [1, 0, 0, 0],  # 1  H
    [2, 0, 0, 0],  # 2  He
    [3, 0, 0, 0],  # 3  Li
    [4, 0, 0, 0],  # 4  Be
    [4, 1, 0, 0],  # 5  B
    [4, 2, 0, 0],  # 6  C
    [4, 3, 0, 0],  # 7  N
    [4, 4, 0, 0],  # 8  O
    [4, 5, 0, 0],  # 9  F
    [4, 6, 0, 0],  # 10  Ne
    [5, 6, 0, 0],  # 11  Na
    [6, 6, 0, 0],  # 12  Mg
    [6, 7, 0, 0],  # 13  Al
    [6, 8, 0, 0],  # 14  Si
    [6, 9, 0, 0],  # 15  P
    [6, 10, 0, 0],  # 16  S
    [6, 11, 0, 0],  # 17  Cl
    [6, 12, 0, 0],  # 18  Ar
    [7, 12, 0, 0],  # 19  K
    [8, 12, 0, 0],  # 20  Ca
    [8, 12, 1, 0],  # 21  Sc
    [8, 12, 2, 0],  # 22  Ti
    [8, 12, 3, 0],  # 23  V
    [8, 12, 4, 0],  # 24  Cr
    [7, 12, 6, 0],  # 25  Mn
    [7, 12, 7, 0],  # 26  Fe
    [7, 12, 8, 0],  # 27  Co
    [7, 12, 9, 0],  # 28  Ni
    [7, 12, 10, 0],  # 29  Cu
    [8, 12, 10, 0],  # 30  Zn
    [8, 13, 10, 0],  # 31  Ga
    [8, 14, 10, 0],  # 32  Ge
    [8, 15, 10, 0],  # 33  As
    [8, 16, 10, 0],  # 34  Se
    [8, 17, 10, 0],  # 35  Br
    [8, 18, 10, 0],  # 36  Kr
    [9, 18, 10, 0],  # 37  Rb
    [10, 18, 10, 0],  # 38  Sr
    [10, 18, 11, 0],  # 39  Y
    [10, 18, 12, 0],  # 40  Zr
    [10, 18, 13, 0],  # 41  Nb
    [9, 18, 15, 0],  # 42  Mo
    [9, 18, 16, 0],  # 43  Tc
    [8, 18, 18, 0],  # 44  Ru
    [8, 18, 19, 0],  # 45  Rh
    [8, 18, 20, 0],  # 46  Pd
    [9, 18, 20, 0],  # 47  Ag
    [10, 18, 20, 0],  # 48  Cd
    [10, 19, 20, 0],  # 49  In
    [10, 20, 20, 0],  # 50  Sn
    [10, 21, 20, 0],  # 51  Sb
    [10, 22, 20, 0],  # 52  Te
    [10, 23, 20, 0],  # 53  I
    [10, 24, 20, 0],  # 54  Xe
    [11, 24, 20, 0],  # 55  Cs
    [12, 24, 20, 0],  # 56  Ba
    [12, 24, 20, 1],  # 57  La
    [12, 24, 20, 2],  # 58  Ce
    [12, 24, 20, 3],  # 59  Pr
    [12, 24, 20, 4],  # 60  Nd
    [12, 24, 20, 5],  # 61  Pm
    [12, 24, 20, 6],  # 62  Sm
    [12, 24, 20, 7],  # 63  Eu
    [12, 24, 20, 8],  # 64  Gd
    [12, 24, 20, 9],  # 65  Tb
    [12, 24, 20, 10],  # 66  Dy
    [12, 24, 20, 11],  # 67  Ho
    [12, 24, 20, 12],  # 68  Er
    [12, 24, 20, 13],  # 69  Tm
    [12, 24, 20, 14],  # 70  Yb
    [12, 24, 21, 14],  # 71  Lu
    [12, 24, 22, 14],  # 72  Hf
    [12, 24, 23, 14],  # 73  Ta
    [11, 24, 25, 14],  # 74  W
    [11, 24, 26, 14],  # 75  Re
    [10, 24, 28, 14],  # 76  Os
    [10, 24, 29, 14],  # 77  Ir
    [10, 24, 30, 14],  # 78  Pt
    [11, 24, 30, 14],  # 79  Au
    [12, 24, 30, 14],  # 80  Hg
    [12, 25, 30, 14],  # 81  Tl
    [12, 26, 30, 14],  # 82  Pb
    [12, 27, 30, 14],  # 83  Bi
    [12, 28, 30, 14],  # 84  Po
    [12, 29, 30, 14],  # 85  At
    [12, 30, 30, 14],  # 86  Rn
    [13, 30, 30, 14],  # 87  Fr
    [14, 30, 30, 14],  # 88  Ra
    [14, 30, 30, 15],  # 89  Ac
    [14, 30, 30, 16],  # 90  Th
    [14, 30, 30, 17],  # 91  Pa
    [13, 30, 30, 19],  # 92  U
    [13, 30, 30, 20],  # 93  Np
    [13, 30, 30, 21],  # 94  Pu
    [13, 30, 30, 22],  # 95  Am
    [13, 30, 30, 23],  # 96  Cm
    [13, 30, 30, 24],  # 97  Bk
    [13, 30, 30, 25],  # 98  Cf
    [13, 30, 30, 26],  # 99  Es
    [13, 30, 30, 27],  # 100  Fm
    [13, 30, 30, 28],  # 101  Md
    [14, 30, 30, 28],  # 102  No
    [14, 30, 31, 28],  # 103  Lr
    [14, 30, 32, 28],  # 104  Rf
    [13, 30, 34, 28],  # 105  Db
    [12, 30, 36, 28],  # 106  Sg
    [12, 30, 37, 28],  # 107  Bh
    [12, 30, 38, 28],  # 108  Hs
    [12, 30, 39, 28],  # 109  Mt
    [12, 30, 40, 28],  # 110  Ds
    [13, 30, 40, 28],  # 111  Rg
    [14, 30, 40, 28],  # 112  Cn
    [14, 31, 40, 28],  # 113  Nh
    [14, 32, 40, 28],  # 114  Fl
    [14, 33, 40, 28],  # 115  Mc
    [14, 34, 40, 28],  # 116  Lv
    [14, 35, 40, 28],  # 117  Ts
    [14, 36, 40, 28],  # 118  Og
]
