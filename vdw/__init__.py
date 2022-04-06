from vdw.util.hirshfeld import HirshfeldAnalysis
import_fail = ""

try:
    from vdw.modeldftd3 import WithDFTD3, to_dftd3
except ImportError:
    import_fail += "conda install simple-dftd3 -c conda-forge\n"
    import_fail += "conda install dftd3-python -c conda-forge\n"

try:
    from vdw.modeldftd4 import WithDFTD4, to_dftd4
except ImportError:
    import_fail += "conda install dftd4 -c conda-forge\n"
    import_fail += "conda install dftd4-python -c conda-forge\n"

try:
    from vdw.modelmbd import WithMBD, to_mbd
except ImportError:
    import_fail += "conda install libmbd -c conda-forge\n"
    import_fail += "pip install pymbd\n"


if import_fail != "":
    import_fail = "Several prelimiary packages not installed. Do me a favour ;)\n" + import_fail
    raise ImportError(import_fail)
