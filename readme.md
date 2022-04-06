# vdw (wrapper of existing vDW packages for PySCF)

This package is a wrapper of various existing vDW libraries for PySCF.

You may need to further install other required packages from conda or pip.

The author is aware of previous efforts to implement vDW for PySCF, such as [pyscf/dftd3](https://github.com/pyscf/dftd3)
and [pyscf/mbd](https://github.com/pyscf/mbd). However, due to my own requirement for usage and API convenience, as well
as my need to use TS-vDW, this simple hundreds-lines-of-code tiny package is built from existing various libraries
for PySCF.

## Included vDW models

### DFTD3

* Package: `simple-dftd3`, https://github.com/awvwgk/simple-dftd3
* Install:
  ```bash
  conda install simple-dftd3 -c conda-forge
  ```
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
* Install:
  ```bash
  conda install dftd4 -c conda-forge
  ```
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

### Many-Body Dispersion and Tkatchenko-Scheffler

* Package: `libmbd`, https://github.com/libmbd/libmbd
* Install:
  ```bash
  conda install libmbd -c conda-forge
  pip install pymbd
  ```

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

* Citations:

  * Tkatchenko-Scheffler

    Tkatchenko, A.; Scheffler, M.
    *Phys. Rev. Lett.* **2009**, *102* (7), 073005. https://doi.org/10.1103/PhysRevLett.102.073005.

  * MBD

    Tkatchenko, A.; DiStasio, R. A.; Car, R.; Scheffler, M.
    *Phys. Rev. Lett.* **2012**, *108* (23), 236402. https://doi.org/10.1103/PhysRevLett.108.236402.

    Ambrosetti, A.; Reilly, A. M.; DiStasio, R. A.; Tkatchenko, A.
    *J. Chem. Phys.* **2014**, *140* (18), 18A508. https://doi.org/10.1063/1.4865104.
  
## More Examples

Refer to [example](example) folder for more examples.
