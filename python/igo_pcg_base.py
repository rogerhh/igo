from common_packages import *

from igo_base import IgoBase

"""
Base class that all PCG type implementations should derive from
Implements incremental_opt() using 
"""
class IgoPCGBase(IgoBase):
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None
        self.chol_A = csr_matrix(([], ([], [])), shape=(0, 0))

    @abstractmethod
    def generate_preconditioner(self, A_tilde, b_tilde, A_hat, b_hat, A_tilde_rows, b_tilde_rows, A_hat_rows, b_hat_rows, params):
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
    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        old_rows, old_cols = self.A.shape
        new_rows, new_cols = A_hat.shape
        resize_cols = new_cols if new_cols > old_cols else old_cols
        resize_rows = new_rows if new_rows > old_rows else old_rows

        # Do all the necessary resizing here
        self.chol_A.resize((resize_rows, resize_cols))
        self.A.resize((resize_rows, resize_cols))
        self.negA.resize((resize_rows, resize_cols))
        self.b.resize((resize_rows, 1))
        self.negb.resize((resize_rows, 1))
        self.diagLamb.resize((resize_cols, resize_cols))
        self.diagLamb[diagLamb.nonzero()] = diagLamb[diagLamb.nonzero()]
        A_tilde.resize((resize_rows, resize_cols))
        b_tilde.resize((resize_rows, 1))

        # 1. Replace all rows in A with corresponding rows of A_tilde
        # 2. Compare A to chol_A and pick high norm rows
        # 3. Add A_hat to A and chol_A
        # 4. Compute L as cholesky_AAt(chol_A.T)
        # 5. Permute columns of A with P
        # 6. Use conjugate gradient to solve A^TAx = A^Tb

        A_tilde_rows = np.array(list(set(A_tilde.nonzero()[0])))
        b_tilde_rows = np.array(list(set(b_tilde.nonzero()[0])))
        A_hat_rows = np.array(list(set(A_hat.nonzero()[0])))
        b_hat_rows = np.array(list(set(b_hat.nonzero()[0])))

        # 1. Replace all rows in A with corresponding rows of A_tilde
        if A_tilde.shape[0] != 0 and A_tilde.shape[1] != 0:
            log("A shapes = ", self.A.shape, len(A_tilde_rows))
            self.A[A_tilde_rows, 0:A_tilde.shape[1]] = A_tilde[A_tilde_rows]
            self.b[b_tilde_rows] = b_tilde[b_tilde_rows]

        self.L, self.P = self.generate_preconditioner(A_tilde, b_tilde, A_hat, b_hat, A_tilde_rows, b_tilde_rows, A_hat_rows, b_hat_rows, params)

        if len(A_hat_rows) > 0:
            self.A[A_hat_rows] = A_hat[A_hat_rows]
            self.b[b_hat_rows] = b_hat[b_hat_rows]

        # 5. Permute columns of A with P
        AP = self.A[:, self.P]
        negAP = self.negA[:, self.P]
        diagLambP = self.diagLamb[self.P][:, self.P]

        log(AP.shape, self.b.shape, negAP.shape, self.negb.shape, diagLambP.shape)
        rhs = AP.T @ self.b - negAP.T @ self.negb
        rhs = spsolve_triangular(self.L, rhs.A, lower=True)
        print(rhs.shape)

        # 6. Use conjugate gradient to solve A^TAx = A^Tb
        linOps = applyPreconditionedATA(AP, negAP, diagLambP, self.L)
        CGcallback.reset()
        LTxP, info = cg(linOps, rhs, callback=CGcallback.callback, tol=1e-10, maxiter=100)
        LTxP = LTxP.reshape((-1, 1))

        xP = spsolve_triangular(self.L.T, LTxP, lower=False)

        self.x = np.zeros_like(xP)
        self.x[self.P] = xP
        
        print(f"cg iter = {CGcallback.count}")


        # self.H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb
        # self.Atb = self.A.T @ self.b

        # self.factor = cholesky(self.H)
        # self.x = self.factor(self.Atb)

        return self.x

    def marginalize(self, cols):
        pass


