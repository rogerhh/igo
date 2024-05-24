from common_packages import *

from igo_row_selector_base import IgoRowSelectorBase

class IgoRowSelectorHighNorm(IgoRowSelectorBase):
    """
    Selects a fixed number of rows with the highest norm
    """

    id_string = "highnorm"

    def __init__(self, params):
        super().__init__(params)

    def select_rows(self, params, **kwargs):
        percent_sel_rows = params["iterative"]["selchol"]["percent_rows"]

        A = kwargs["A"]
        C = kwargs["C"]
        sqrtLamb = kwargs["sqrtLamb"]

        chol_A = kwargs["chol_A"]
        chol_C = kwargs["chol_C"]
        chol_sqrtLamb = kwargs["chol_sqrtLamb"]

        A_diff = chol_A - A
        C_diff = chol_C - C
        sqrtLamb_diff = chol_sqrtLamb - sqrtLamb

        A_diff_row_norms = scipy.sparse.linalg.norm(A_diff, ord=float('inf'), axis=1)
        C_diff_row_norms = scipy.sparse.linalg.norm(C_diff, ord=float('inf'), axis=1)
        sqrtLamb_diff_row_norms = scipy.sparse.linalg.norm(sqrtLamb_diff, ord=float('inf'), axis=1)

        A_rows = A.shape[0]
        C_rows = C.shape[0]
        sqrtLamb_rows = sqrtLamb.shape[0]

        diff_row_norms = []
        diff_row_norms.extend(A_diff_row_norms)
        diff_row_norms.extend(C_diff_row_norms)
        diff_row_norms.extend(sqrtLamb_diff_row_norms)
        diff_row_norms = np.array(diff_row_norms)

        num_diff = len(np.where(diff_row_norms > 0)[0])

        num_sel_rows = int(diff_row_norms.shape[0] * percent_sel_rows)

        high_rows = np.argsort(diff_row_norms)[-num_sel_rows:]

        A_sel_rows = []
        C_sel_rows = []
        sqrtLamb_sel_rows = []

        for row in high_rows:
            if row < A_rows:
                A_sel_rows.append(row)
            elif row < A_rows + C_rows:
                C_sel_rows.append(row - A_rows)
            else:
                sqrtLamb_sel_rows.append(row - A_rows - C_rows)

        return A_sel_rows, C_sel_rows, sqrtLamb_sel_rows
