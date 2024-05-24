"""
linear_operator.py: A subclass of scipy.linalg.LinearOperator. Given A, L, implements matvec(v) that returns L^-1A^TAL^-Tv, but applies each matrix to v sequentially
"""

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg, spsolve_triangular, inv
import numpy as np

class applyATA(LinearOperator):
    def __init__(self, A, C, sqrtLamb):
        super().__init__(shape=(A.shape[1], A.shape[1]), dtype="double")
        assert(A.shape[1] == C.shape[1])
        assert(A.shape[1] == sqrtLamb.shape[1])
        self.A = A
        self.C = C
        self.sqrtLamb = sqrtLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.C_rows = C.shape[0]

    # This performs (A^T A - C^T C + sqrtLamb^T sqrtLamb)v
    def _matvec(self, v):

        v2_A = self.A @ v
        v2_C = self.C @ v
        v2_sqrtLamb = self.sqrtLamb @ v

        v3 = self.A.T @ v2_A - self.C.T @ v2_C + self.sqrtLamb.T @ v

        return v3
    
    def _matmat(self, V):
        V2_A = self.A @ V
        V2_C = self.C @ V
        V2_sqrtLamb = self.sqrtLamb @ V

        V3 = self.A.T @ V2_A - self.C.T @ V2_C + self.sqrtLamb.T @ V

        return V3

class applyPreconditionedATA(LinearOperator):
    def __init__(self, A, C, sqrtLamb, L):
        super().__init__(shape=(A.shape[1], A.shape[1]), dtype="double")
        assert(A.shape[1] == C.shape[1])
        assert(A.shape[1] == sqrtLamb.shape[1])
        self.L = L
        self.A = A
        self.C = C
        self.sqrtLamb = sqrtLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.C_rows = C.shape[0]

    # This performs L^-1(A^TA - C^TC + sqrtLamb^TsqrtLamb)L^-Tv
    def _matvec(self, v):

        v1 = spsolve_triangular(self.L.T, v, lower=False)

        v2_A = self.A @ v1
        v2_C = self.C @ v1
        v2_sqrtLamb = self.sqrtLamb @ v1

        v3 = self.A.T @ v2_A - self.C.T @ v2_C + self.sqrtLamb.T @ v2_sqrtLamb

        v4 = spsolve_triangular(self.L, v3, lower=True)

        return v4

    # This performs L^-1(A^TA - C^TC + sqrtLamb^TsqrtLamb)L^-TV
    def _matmat(self, V):
        V1 = spsolve_triangular(self.L.T, V, lower=False)

        V2_A = self.A @ V1
        V2_C = self.C @ V1
        V2_sqrtLamb = self.sqrtLamb @ V1

        V3 = self.A.T @ V2_A - self.C.T @ V2_C + self.sqrtLamb.T @ V1

        V4 = spsolve_triangular(self.L, V3, lower=True)

        return V4

# Computes [vA \\ ivC \\ vLamb] = [A \\ iC \\ sqrtLamb]L^-Tv or L^-1 (A^T vA + iC^T ivC + sqrtLamb^T vLamb)
class applyPreconditionedA(LinearOperator):
    def __init__(self, A, C, sqrtLamb, L):
        super().__init__(shape=(A.shape[0], A.shape[1]), dtype="double")
        assert(A.shape[1] == C.shape[1])
        assert(A.shape[1] == sqrtLamb.shape[1])
        self.L = L
        self.A = A
        self.C = C
        self.sqrtLamb = sqrtLamb
        self.A_rows = A.shape[0]
        self.A_cols = A.shape[1]
        self.C_rows = C.shape[0]
        self.sqrtLamb_rows = sqrtLamb.shape[0]

    # This performs [vA \\ ivC] = [A \\ iC]L^-Tv
    def _matvec(self, v):

        v1 = spsolve_triangular(self.L.T, v, lower=False)

        v2 = np.zeros((self.A_rows + self.C_rows + self.sqrtLamb_rows, 1))
        v2[:self.A_rows, 0] = self.A @ v1
        v2[self.A_rows:self.A_rows + self.C_rows, 0] = self.C @ v1
        v2[self.A_rows + self.C_rows:, 0] = self.sqrtLamb @ v1

        return v2

    # This performs L^-1 (A^T vA + iC^T ivC + sqrtLamb^T vLamb)
    def _rmatvec(self, v):
        v1 = self.A.T @ v[:self.A_rows] \
                - self.C.T @ v[self.A_rows:self.A_rows + self.C_rows] \
                + self.sqrtLamb.T @ v[self.A_rows + self.C_rows:]

        v2 = spsolve_triangular(self.L, v1, lower=True)

        return v2

    # This performs [VA \\ iVC] = [A \\ iC]L^-TV
    def _matmat(self, V):
        V1 = spsolve_triangular(self.L.T, V, lower=False)

        V2 = np.zeros((self.A_rows + self.C_rows + self.sqrtLamb_rows, V.shape[1]))
        V2[:self.A_rows] = self.A @ V1
        V2[self.A_rows:self.A_rows + self.C_rows] = self.C @ V1
        V2[self.A_rows + self.C_rows:] = self.sqrtLamb @ V1

        return V2

    # This performs L^-1 (A^T VA + iC^T iVC + sqrtLamb^T VLamb)
    def _rmatmat(self, V):
        V1 = self.A.T @ V[:self.A_rows] \
                - self.C.T @ V[self.A_rows:self.A_rows + self.C_rows] \
                + self.sqrtLamb.T @ V[self.A_rows + self.C_rows:]

        V2 = spsolve_triangular(self.L, V1, lower=True)

        return V2

