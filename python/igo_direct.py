from common_packages import *

from igo_base import IgoBase

class IgoDirect(IgoBase):
    """
    Compute the optimal solution for the incremental optimization problem using direct methods
    """

    id_string = "direct"

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

        # Insert the new values
        self.A[A_hat.nonzero()] = A_hat[A_hat.nonzero()]
        self.b[b_hat.nonzero()] = b_hat[b_hat.nonzero()]
        self.C[C_hat.nonzero()] = C_hat[C_hat.nonzero()]
        self.d[d_hat.nonzero()] = d_hat[d_hat.nonzero()]
        self.sqrtLamb[sqrtLamb_hat.nonzero()] = sqrtLamb_hat[sqrtLamb_hat.nonzero()]

        # Compute the solution
        H = self.A.T @ self.A - self.C.T @ self.C + self.sqrtLamb.T @ self.sqrtLamb
        Atb = self.A.T @ self.b - self.C.T @ self.d

        factor = cholesky(H)
        L = factor.L()
        P = factor.P()
        x = factor(Atb)
    
        return x
    
    def marginalize(self, keys):
        """
        Marginalize out the variables in keys
        """
        raise NotImplementedError

