from common_packages import *

"""
Base class that all igo implementations should derive from. 
Implements incremental_opt() and marginalize()
"""
class IgoBase(ABC):

    def __init__(self, params):
        self.A = csr_matrix(([], ([], [])), shape=(0, 0))
        self.b = csr_matrix(([], ([], [])), shape=(0, 0))
        self.negA = csr_matrix(([], ([], [])), shape=(0, 1))
        self.negb = csr_matrix(([], ([], [])), shape=(0, 1))

        self.diagLamb = csr_matrix(([], ([], [])), shape=(0, 0))

    @abstractmethod
    def setup_lc_step(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        raise NotImplementedError

    @abstractmethod
    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        raise NotImplementedError

    @abstractmethod
    def marginalize(self, keys):
        raise NotImplementedError

    """
    This function is defined a bit differently than in the paper. Here, A_tilde directly replaces 
    coressponding entries in A
    Implements the following: 
        A[nonzero(A_tilde)] = A_tilde
        A = [A; A_hat]
        b[nonzero(b_tilde)] = b_tilde
        b = [b; b_hat]
        return solve(argmin_x \|Ax - b\|_2^2)
    """
    def baseline_solve(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        self.A[A_tilde.nonzero()] = A_tilde[A_tilde.nonzero()]
        self.b[b_tilde.nonzero()] = b_tilde[b_tilde.nonzero()]

        old_rows, old_cols = self.A.shape
        new_rows, new_cols = A_hat.shape

        resize_cols = new_cols if new_cols > old_cols else old_cols
        resize_rows = new_rows if new_rows > old_rows else old_rows

        self.A.resize((resize_rows, resize_cols))
        self.negA.resize((resize_rows, resize_cols))
        self.b.resize((resize_rows, 1))
        self.negb.resize((resize_rows, 1))

        self.diagLamb.resize((resize_cols, resize_cols))
        self.diagLamb[diagLamb.nonzero()] = diagLamb[diagLamb.nonzero()]

        self.A[A_hat.nonzero()] = A_hat[A_hat.nonzero()]
        self.b[b_hat.nonzero()] = b_hat[b_hat.nonzero()]

        self.H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb.T @ self.diagLamb
        self.Atb = self.A.T @ self.b

        factor = cholesky(self.H)
        self.L = factor.L()
        self.P = factor.P()
        self.x = factor(self.Atb)

        return self.x

