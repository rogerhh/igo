"""
linear_operator.py: A subclass of scipy.linalg.LinearOperator. Given A, L, implements matvec(v) that returns L^-1A^TAL^-Tv, but applies each matrix to v sequentially
"""

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, spsolve_triangular, inv

class applyATA(LinearOperator):
    def __init__(self, A, negA, sqrtLamb, L):
        super().__init__(shape=(L.shape), dtype="double")
        assert(A.shape[1] == negA.shape[1])
        assert(A.shape[1] == sqrtLamb.shape[0])
        assert(A.shape[1] == sqrtLamb.shape[1])
        self.L = L
        self.A = A
        self.negA = negA
        self.sqrtLamb = sqrtLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.negA_rows = negA.shape[0]

    # This performs ATAv
    def _matvec(self, v):

        v1 = spsolve_triangular(self.L.T, v, lower=False)

        v2_A = self.A @ v1
        v2_negA = self.negA @ v1
        v2_lamb = self.sqrtLamb @ v1

        v3 = self.A.T @ v2_A - self.negA.T @ v2_negA + self.sqrtLamb.T @ v2_lamb

        v4 = spsolve_triangular(self.L, v3, lower=True)

        return v4

