from common_packages import *

from igo_sel_chol_base import IgoSelCholBase

class IgoSelectiveCholeskyUpdate4(IgoSelCholBase):
    """
    SelCholUpdate4 selects the rows that affects the difference in Hessian the most
    """
    id_string = "selcholupdate4"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None
        self.chol_A = csr_matrix(([], ([], [])), shape=(0, 0))

    """
    Run incremental_opt until convergence, saving internal state of the incremental solver
    while doing so
    """
    def setup_lc_step(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        return super().setup_lc_step(A_tilde, b_tilde, A_hat, b_hat, diagLamb, params)

    def select_rows(self, A_tilde, b_tilde, A_hat, b_hat, params):
        abs_A_diff = abs(self.chol_A - self.A)
        abs_chol_A = abs(self.chol_A)
        abs_A = abs(self.A)
        A_diff_sum = abs_A_diff.sum(axis=1)
        chol_A_sum = abs_chol_A.sum(axis=1)

        A_diff_rows = np.array(list(set(abs_A_diff.nonzero()[0])))
        if "setup_lc_step" in params.keys() and params["setup_lc_step"]:
            return A_diff_rows

        # 2-norm < sqrt(nnz_col) * infty norm < thresh
        # Count nnz_col here
        old_size = self.H.shape[1]
        col_nnz_sqrt = np.zeros((old_size))
        for j in range(self.H.shape[1]):
            nnz = len(self.H[:, j].nonzero()[1])
            col_nnz_sqrt[j] = math.sqrt(nnz) if nnz != 0 else 1e-12
            assert(nnz != 0)

        selected_rows = []
        row_importance = np.zeros((self.A.shape[0]))

        for i in A_diff_rows:
            s = abs_A_diff[i] * chol_A_sum[i, 0]  + abs_A[i] * A_diff_sum[i, 0]
            j_indices = s.nonzero()[1]
            s[0, j_indices] = s[0, j_indices] # * col_nnz_sqrt[j_indices]
            row_importance[i] = s.max()
            # print(s)
            # print(row_importance[i])

        # if len(A_diff_rows):
        #     exit(0)

        max_rows = self.A.shape[0]
        percent_rows = params["selchol"]["percent_rows"]
        num_rows = max(int(percent_rows * len(A_diff_rows)), params["selchol"]["min_sel_rows"])
        high_rows = np.argsort(row_importance)[max_rows-num_rows:max_rows]


        return high_rows


        diff_row_norms = scipy.sparse.linalg.norm(A_diff, ord=float('inf'), axis=1)
        A_row_norms = scipy.sparse.linalg.norm(self.A, ord=float('inf'), axis=1)

        diff_threshold = params[self.id_string]["diff_threshold"]
        # high_rows = np.where(diff_row_norms >= diff_threshold)[0]
        high_rows = np.where(diff_row_norms >= diff_threshold)[0]
        # return high_rows
