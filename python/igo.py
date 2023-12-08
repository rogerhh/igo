from igo_base import IgoBase
from igo_baseline import IgoBaseline
from igo_sel_chol_update import IgoSelectiveCholeskyUpdate
from igo_sel_chol_update2 import IgoSelectiveCholeskyUpdate2
from igo_sel_chol_update3 import IgoSelectiveCholeskyUpdate3
from igo_bounded_diff import IgoBoundedDiff

igo_types = [IgoBaseline, \
             IgoSelectiveCholeskyUpdate, \
             IgoSelectiveCholeskyUpdate2, \
             IgoSelectiveCholeskyUpdate3, \
             IgoBoundedDiff] # , \
             # IgoIncompleteCholeskyStrongConnection, \
             # IgoIncompleteCholeskyValueFilter]

