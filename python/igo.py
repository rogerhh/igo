from igo_base import IgoBase
from igo_baseline import IgoBaseline
from igo_sel_chol_update import IgoSelectiveCholeskyUpdate
from igo_sel_chol_update2 import IgoSelectiveCholeskyUpdate2
from igo_sel_chol_update3 import IgoSelectiveCholeskyUpdate3
from igo_sel_chol_update4 import IgoSelectiveCholeskyUpdate4
from igo_bounded_diff import IgoBoundedDiff
from igo_bounded_diff2 import IgoBoundedDiff2
from igo_extended_diagonal import IgoExtendedDiagonal
from igo_identity import IgoIdentity

igo_types = [IgoBaseline, \
             IgoSelectiveCholeskyUpdate, \
             IgoSelectiveCholeskyUpdate2, \
             IgoSelectiveCholeskyUpdate3, \
             IgoSelectiveCholeskyUpdate4, \
             IgoBoundedDiff, \
             IgoBoundedDiff2, \
             IgoExtendedDiagonal, \
             IgoIdentity] #, \
             # IgoIncompleteCholeskyStrongConnection, \
             # IgoIncompleteCholeskyValueFilter]

