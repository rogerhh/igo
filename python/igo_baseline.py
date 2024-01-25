from common_packages import *

from igo_base import IgoBase

class IgoBaseline(IgoBase):
    """
    The baseline Igo is only used for correctness checks and is very inefficient. 
    """
    id_string = "baseline"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None

    def setup_lc_step(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        setup_params = params
        return self.incremental_opt(A_tilde, b_tilde, A_hat, b_hat, diagLamb, setup_params)

    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        return self.baseline_solve(A_tilde, b_tilde, A_hat, b_hat, diagLamb, params)

