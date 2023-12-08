from common_packages import *

from igo_sel_chol_base import IgoSelCholBase

class IgoBoundedDiff(IgoSelCholBase):
    """
    BoundedDiff is for profiling. Picks all rows whose norm exceeds a threshold
    """
    id_string = "boundeddiff"
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
        setup_params = deepcopy(params)
        setup_params["boundeddiff"]["diff_threshold"] = 0
        return self.incremental_opt(A_tilde, b_tilde, A_hat, b_hat, diagLamb, setup_params)

    def select_rows(self, params):
        A_diff = self.chol_A - self.A
        diff_row_norms = scipy.sparse.linalg.norm(A_diff, ord=float('inf'), axis=1)
        A_row_norms = scipy.sparse.linalg.norm(self.A, ord=float('inf'), axis=1)

        diff_threshold = params[self.id_string]["diff_threshold"]
        # high_rows = np.where(diff_row_norms >= diff_threshold)[0]
        high_rows = np.where(diff_row_norms * A_row_norms >= diff_threshold)[0]
        return high_rows
