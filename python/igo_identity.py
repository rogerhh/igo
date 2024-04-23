from common_packages import *

from igo_preconditioner_generator_base import IgoPreconditionerGeneratorBase

class IgoIdentity(IgoPreconditionerGeneratorBase):
    """
    No preconditioner
    """
    id_string = "identity"

    def __init__(self, params):
        super.__init__()

    def generate(self, igo, old_cols, new_cols, A_updated_rows, A_new_rows, params):
        M = scipy.sparse.eye(new_cols, new_cols)
        P = np.arange(new_cols)
        return M, P


# from igo_pcg_base import IgoPCGBase
# from igo_baseline import IgoBaseline

# class IgoIdentity(IgoPCGBase):
#     """
#     No preconditioner
#     """
#     id_string = "identity"
#     def __init__(self, params):
#         super().__init__(params)
# 
#     """
#     Run incremental_opt until convergence, saving internal state of the incremental solver
#     while doing so
#     """
#     def setup_lc_step(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
#         return self.baseline_solve(A_tilde, b_tilde, A_hat, b_hat, diagLamb, params)
# 
#     def generate_preconditioner(self, A_tilde, b_tilde, A_hat, b_hat, A_tilde_rows, b_tilde_rows, A_hat_rows, b_hat_rows, params):
#         old_size = self.L.shape[0]
#         new_size = self.A.shape[1]
#         self.L = scipy.sparse.eye(new_size, new_size)
# 
#         new_indices = np.arange(old_size, new_size)
#         self.P = deepcopy(self.P)
#         self.P.resize(new_size)
#         self.P[new_indices] = new_indices
#         return self.L, self.P
