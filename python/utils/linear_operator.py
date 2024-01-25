"""
linear_operator.py: A subclass of scipy.linalg.LinearOperator. Given A, L, implements matvec(v) that returns L^-1A^TAL^-Tv, but applies each matrix to v sequentially
"""

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, spsolve_triangular, inv

class applyATA(LinearOperator):
    def __init__(self, A, negA, diagLamb):
        super().__init__(shape=(A.shape[1], A.shape[1]), dtype="double")
        assert(A.shape[1] == negA.shape[1])
        assert(A.shape[1] == diagLamb.shape[0])
        assert(A.shape[1] == diagLamb.shape[1])
        self.A = A
        self.negA = negA
        self.diagLamb = diagLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.negA_rows = negA.shape[0]

    # This performs ATAv
    def _matvec(self, v):

        v2_A = self.A @ v
        v2_negA = self.negA @ v

        v3 = self.A.T @ v2_A - self.negA.T @ v2_negA + self.diagLamb.T @ v

        return v3

    def _matmat(self, V):
        print("matmat")
        V2_A = self.A @ V
        V2_negA = self.negA @ V

        V3 = self.A.T @ V2_A - self.negA.T @ V2_negA + self.diagLamb @ V

        return V3

class applyPreconditionedATA(LinearOperator):
    def __init__(self, A, negA, diagLamb, L):
        super().__init__(shape=(A.shape[1], A.shape[1]), dtype="double")
        assert(A.shape[1] == negA.shape[1])
        assert(A.shape[1] == diagLamb.shape[0])
        assert(A.shape[1] == diagLamb.shape[1])
        self.L = L
        self.A = A
        self.negA = negA
        self.diagLamb = diagLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.negA_rows = negA.shape[0]

    # This performs ATAv
    def _matvec(self, v):

        v1 = spsolve_triangular(self.L.T, v, lower=False)

        v2_A = self.A @ v1
        v2_negA = self.negA @ v1

        v3 = self.A.T @ v2_A - self.negA.T @ v2_negA + self.diagLamb @ v1

        v4 = spsolve_triangular(self.L, v3, lower=True)

        return v4

    def _matmat(self, V):
        V1 = spsolve_triangular(self.L.T, V, lower=False)

        V2_A = self.A @ V1
        V2_negA = self.negA @ V1

        V3 = self.A.T @ V2_A - self.negA.T @ V2_negA + self.diagLamb * V

        V4 = spsolve_triangular(self.L, V3, lower=True)

