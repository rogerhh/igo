from common_packages import *

from igo_pcg_base import IgoPCGBase

class IgoSelCholBase(IgoPCGBase):
    """
    Base class for all selective cholesky update preconditioning.
    Pick a set of rows in A to replace in chol_A
    """
    id_string = "boundeddiff"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None
        self.chol_A = csr_matrix(([], ([], [])), shape=(0, 0))

    """
    Select rows of A to replace in chol_A
    """
    @abstractmethod
    def select_rows(self, params):
        raise NotImplementedError

    def generate_preconditioner(self, A_tilde, b_tilde, A_hat, b_hat, A_tilde_rows, b_tilde_rows, A_hat_rows, b_hat_rows, params):

        selected_rows = self.select_rows(params)

        log("num relin_rows = ", len(selected_rows))

        max_diff = abs(self.A - self.chol_A).max()
        h_max_diff = abs(self.A.T @ self.A - self.chol_A.T @ self.chol_A).max()
        log(f"max_diff = {max_diff}, h_max_diff = {h_max_diff}")

        if len(selected_rows) > 0:
            self.chol_A[selected_rows] = self.A[selected_rows]

        max_diff = abs(self.A - self.chol_A).max()
        h_max_diff = abs(self.A.T @ self.A - self.chol_A.T @ self.chol_A).max()
        log(f"After update, max_diff = {max_diff}, h_max_diff = {h_max_diff}")

        # 3. Add A_hat to A and chol_A
        if len(A_hat.nonzero()[0]) > 0:
            self.chol_A[A_hat_rows] = A_hat[A_hat_rows]

        # 4. Compute L as cholesky_AAt(chol_A.T)
        self.H = self.chol_A.T @ self.chol_A - self.negA.T @ self.negA + self.diagLamb.T @ self.diagLamb
        factor = cholesky(self.H)
        self.L = factor.L()
        self.P = factor.P()

        return self.L, self.P
