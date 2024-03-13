from common_packages import *

from igo_lsqr_sel_chol_base import IgoLSQRSelCholBase

class IgoLSQRSelectiveCholeskyUpdate2(IgoLSQRSelCholBase):
    """
    SelectiveCholeskyUpdate picks the high norm rows to update Cholesky with and use 
    as a preconditioner in LSQR
    v2 is the correct version. We need to keep a separate copy of A to check
    for the rows we incoprated in L
    """
    id_string = "lsqrselcholupdate2"
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

        # 2. Compare A to chol_A and pick high norm rows
        A_diff = self.chol_A - self.A
        diff_row_norms = scipy.sparse.linalg.norm(A_diff, ord=float('inf'), axis=1)
        A_row_norms = scipy.sparse.linalg.norm(self.A, ord=float('inf'), axis=1)

        num_diff = len(np.where(diff_row_norms > 0)[0])

        max_rows = self.A.shape[0]
        percent_rows = params["selchol"]["percent_rows"]
        num_rows = max(int(percent_rows * num_diff), params["selchol"]["min_sel_rows"])
        high_rows = np.argsort(diff_row_norms)[max_rows-num_rows:max_rows]

        return high_rows

