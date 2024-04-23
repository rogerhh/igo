from common_packages import *

from igo_base import IgoBase

class IgoIterative(IgoBase):
    """
    Compute the optimal solution for the incremental optimization problem using iterative methods
    Iterative solvers should have a precond_gen and a solver
    precond_gen genera
    """

    id_string = "iterative"

    def __init__(self, params):
        super().__init__(params)
    
    def incremental_opt(self, 
                        A_tilde, b_tilde, 
                        A_hat, b_hat, 
                        C_tilde, d_tilde,
                        C_hat, d_hat,
                        sqrtLamb_tilde,
                        sqrtLamb_hat, 
                        params):

        old_A_rows, old_A_cols = self.A.shape
        new_A_rows, new_A_cols = A_hat.shape
        old_C_rows, old_C_cols = self.C.shape
        new_C_rows, new_C_cols = C_hat.shape
        old_sqrtLamb_rows, old_sqrtLamb_cols = self.sqrtLamb.shape
        new_sqrtLamb_rows, new_sqrtLamb_cols = sqrtLamb_hat.shape

        assert old_A_cols == old_C_cols == old_sqrtLamb_cols
        assert new_A_cols == new_C_cols == new_sqrtLamb_cols
        assert old_sqrtLamb_rows == old_sqrtLamb_cols
        assert new_sqrtLamb_rows == new_sqrtLamb_cols

        # Resize the matrices to accomodate the new values
        resize_cols = new_A_cols if new_A_cols > old_A_cols else old_A_cols
        resize_rows = new_A_rows if new_A_rows > old_A_rows else old_A_rows

        self.A.resize((resize_rows, resize_cols))
        self.b.resize((resize_rows, 1))
        self.C.resize((resize_rows, resize_cols))
        self.d.resize((resize_rows, 1))
        self.sqrtLamb.resize((resize_cols, resize_cols))

        # Replace entries in A, b, C, d, sqrtLamb with the new values
        self.A[A_tilde.nonzero()] = A_tilde[A_tilde.nonzero()]
        self.b[b_tilde.nonzero()] = b_tilde[b_tilde.nonzero()]
        self.C[C_tilde.nonzero()] = C_tilde[C_tilde.nonzero()]
        self.d[d_tilde.nonzero()] = d_tilde[d_tilde.nonzero()]
        self.sqrtLamb[sqrtLamb_tilde.nonzero()] = sqrtLamb_tilde[sqrtLamb_tilde.nonzero()]


