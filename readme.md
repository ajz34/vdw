# vdw (my naïve wrapper of existing vDW packages for PySCF)

This package is my naïve wrapper of various existing vDW libraries for PySCF. Should be able to evaluate energy or force
(gradient) of van der Waals correction to density functional methods.

This package is not aimed to be a pyscf extension module. It's just a wrapper.

## Install

To install this package, you may download from pypi:

```bash
pip install pyvdw
```

To use DFTD3, DFTD4, or TS-vDW (from libmbd) or MBD methods, you may also manually install those libraries.
This package is only an interface to those existing libraries.

```bash
conda install simple-dftd3 dftd3-python dftd4 dftd4-python libmbd -c conda-forge
pip install pyscf pymbd
```

I know that leaving the task of dependency packages installation to user is really inconvenient, but currently
I don't know how to handle conda, pip and advanced packaging elegently. So if any pratical ideas on this, raise
your issue >w<

## Included vDW models

### DFTD3

* Package: `simple-dftd3`, https://github.com/awvwgk/simple-dftd3

* Usage:
  ```python
  from pyscf import gto, dft
  from vdw import to_dftd3
  mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ").build()
  # Modified/Revisited BJ/Rational damping
  mf = to_dftd3(dft.RKS(mol, xc="PBE"), version="bjm").run()
  print(mf.e_vdw)  # -0.000347885
  ```

* Versions and Citations:
      
  For any version of DFTD3, please first cite 10.1063/1.3382344. This is not formal citation recommendation,
  thus refer to original site [DFTD3](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/)
  and package [simple-dftd3](https://github.com/awvwgk/simple-dftd3) for formal citation guide.
      
  * Original DFTD3 (`version = "zero"`):
      
    Grimme, S.; Antony, J.; Ehrlich, S.; Krieg, H. *J. Chem. Phys.* **2010**, *132* (15), 154104.
    https://doi.org/10.1063/1.3382344.
    
    Goerigk, L.; Hansen, A.; Bauer, C.; Ehrlich, S.; Najibi, A.; Grimme, S.
    *Phys. Chem. Chem. Phys.* **2017**, *19* (48), 32184–32215. https://doi.org/10.1039/C7CP04913G.

  * BJ/Rational damping (`version = "bj"`, which is default):
        
    Grimme, S.; Ehrlich, S.; Goerigk, L. *J. Comput. Chem.* **2011**, *32* (7), 1456–1465.
    https://doi.org/10.1002/jcc.21759.
        
  * Modified zero damping and BJ damping (`version = "zerom" or "bjm"`):

    Smith, D. G. A.; Burns, L. A.; Patkowski, K.; Sherrill, C. D.
    *J. Phys. Chem. Lett.* **2016**, *7* (12), 2197–2203. https://doi.org/10.1021/acs.jpclett.6b00780.

  * Optimized Power (`version = "op"`):
    
    Witte, J.; Mardirossian, N.; Neaton, J. B.; Head-Gordon, M.
    *J. Chem. Theory Comput.* **2017**, *13* (5), 2043–2052. https://doi.org/10.1021/acs.jctc.7b00176.

### DFTD4

* Package: `dftd4`, https://github.com/dftd4/dftd4

* Usage:
  ```python
  from pyscf import gto, dft
  from vdw import to_dftd4
  mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVDZ").build()
  # DFTD4 default bj-eeq-atm version
  mf = to_dftd4(dft.RKS(mol, xc="PBE")).run()
  print(mf.e_vdw)  # -0.000192782
  ```

* Citations:
      
  For any version of DFTD4, please first cite 10.1063/1.5090222. This is not formal citation recommendation,
  thus refer to package [dftd4](https://github.com/dftd4/dftd4) for formal citation guide.

  * Original DFTD4:
    
    Caldeweyher, E.; Ehlert, S.; Hansen, A.; Neugebauer, H.; Spicher, S.; Bannwarth, C.; Grimme, S.
    *J. Chem. Phys.* **2019**, *150* (15), 154122. https://doi.org/10.1063/1.5090222.

  * Newly SCAN related functionals:

    Ehlert, S.; Huniar, U.; Ning, J.; Furness, J. W.; Sun, J.; Kaplan, A. D.; Perdew, J. P.; Brandenburg, J. G.
    *J. Chem. Phys.* **2021**, *154* (6), 061101. https://doi.org/10.1063/5.0041008.

    Bursch, M.; Neugebauer, H.; Ehlert, S.; Grimme, S.
    *J. Chem. Phys.* **2022**, *156* (13), 134105. https://doi.org/10.1063/5.0086040.

  * Doubly hybrid functionals:

    Santra, G.; Sylvetsky, N.; Martin, J. M. L.
    *J. Phys. Chem. A* **2019**, *123* (24), 5129–5143. https://doi.org/10.1021/acs.jpca.9b03157.

### TS and MBD

* Package: `libmbd`, https://github.com/libmbd/libmbd

* Usage:
  ```python
  from pyscf import gto, dft
  from vdw import to_mbd
  mol = gto.Mole(atom="""
      O  0.  0.  0.
      H  0.  0.  1.
      H  0.  1.  0.
      O  0.  0.  2.
      H  0.  0.  3.
      H  0.  1.  2.""", basis="cc-pVDZ").build()
  # Tkatchenko-Scheffler 
  mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="ts").run()
  print(mf.e_vdw)  # -0.000212847
  # MBD@rsSCS
  mf = to_mbd(dft.RKS(mol, xc="PBE"), variant="rsscs").run()
  print(mf.e_vdw)  # -0.001245831
  ```

* Notice

  To calculate MBD or TS-vDW, free atomic volume is required. This is calculated, instead of preloaded,
  using basis set aug-cc-pVQZ. Value of this volume may be close to FHI-aims and Quantum Espresso. However,
  this calculation is relatively costly if your molecule and basis set is not large. Basis set error also
  occurs (where FHI-aims give free folume by highly efficient numerical radial Schrödinger equation).

* Citations:

    This is not formal citation recommendation

  * TS (Tkatchenko-Scheffler)

    Tkatchenko, A.; Scheffler, M.
    *Phys. Rev. Lett.* **2009**, *102* (7), 073005. https://doi.org/10.1103/PhysRevLett.102.073005.

  * MBD (Many-Body Dispersion)

    Tkatchenko, A.; DiStasio, R. A.; Car, R.; Scheffler, M.
    *Phys. Rev. Lett.* **2012**, *108* (23), 236402. https://doi.org/10.1103/PhysRevLett.108.236402.

    Ambrosetti, A.; Reilly, A. M.; DiStasio, R. A.; Tkatchenko, A.
    *J. Chem. Phys.* **2014**, *140* (18), 18A508. https://doi.org/10.1063/1.4865104.
  
## More Examples

Refer to [example](example) folder for more examples.

## Code Sources

This package uses or modifies existing codes.
* Hirshfeld analysis utilizes atomic spherically averaged DFT [pyscf/pyscf #1143](https://github.com/pyscf/pyscf/pull/1143).
* Wrapper code utilizes [pyscf/dftd3](https://github.com/pyscf/dftd3).
* Functional default parameters of TS-vDW and MBD are from [libmbd/libmbd](https://github.com/libmbd/libmbd).

The author is aware of previous efforts to implement vDW for PySCF, such as [pyscf/dftd3](https://github.com/pyscf/dftd3)
and [pyscf/mbd](https://github.com/pyscf/mbd). However, due to my own requirement for usage and API convenience, as well
as my need to use TS-vDW, this simple hundreds-lines-of-code tiny package is built from existing various libraries.
