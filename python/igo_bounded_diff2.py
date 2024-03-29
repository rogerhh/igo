from common_packages import *

from igo_sel_chol_base import IgoSelCholBase

class IgoBoundedDiff2(IgoSelCholBase):
    """
    BoundedDiff2 adds up the difference in hessian to each diagonal
    """
    id_string = "boundeddiff2"
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
        old_row_norm_bound = params["boundeddiff"]["max_norm_bound"]
        params["boundeddiff"]["max_norm_bound"] = 0
        res = self.incremental_opt(A_tilde, b_tilde, A_hat, b_hat, diagLamb, params)
        params["boundeddiff"]["max_norm_bound"] = old_row_norm_bound
        return res

    def select_rows(self, A_tilde, b_tilde, A_hat, b_hat, params):
        diff_norm_sums = np.zeros((self.A.shape[1]))
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

        thresh = params["boundeddiff"]["max_norm_bound"]

        selected_rows = []

        # Use a random permutation to make sure that the first few rows always pass through
        for i in np.random.permutation(A_diff_rows):
            s = abs_A_diff[i] * chol_A_sum[i, 0]  + abs_A[i] * A_diff_sum[i, 0]
            j_indices = s.nonzero()[1]
            diff_norm_sums[j_indices] += s[0, j_indices]
            for j in j_indices:
                if diff_norm_sums[j] * col_nnz_sqrt[j] > thresh:
                    selected_rows.append(i)
                    diff_norm_sums[j_indices] -= s[0, j_indices]
                    break


        selected_rows = np.array(sorted(selected_rows))


        return selected_rows

