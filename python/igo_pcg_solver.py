from common_packages import *

from utils.linear_operator import applyPreconditionedATA
from igo_iterative_solver_base import IgoIterativeSolverBase

class IgoPcgSolver(IgoIterativeSolverBase):
    """
    Solves the linear system using the preconditioned conjugate gradient method
    We are solving the preconditioned linear system 
    M^{-1}(A^T A - C^T C + sqrtLamb^T sqrtLamb)x = M^{-1}(A^T b - C^T d)
    """

    id_string = "pcg"

    def __init__(self, params):
        super().__init__(params)
        self.cg_count = 0

    def cg_count_reset(self):
        self.cg_count = 0

    def cg_callback(self, x):
        self.cg_count += 1

    def solve(self, A, b, C, d, sqrtLamb, M, P, params):
        # Permute columns of A, C, sqrtLamb with P
        AP = A[:, P]
        CP = C[:, P]
        sqrtLambP = sqrtLamb[:, P]

        # Compute the right-hand side of the normal equations
        rhs = AP.T @ b - CP.T @ d
        rhs = spsolve_triangular(M, rhs.A, lower=True)

        # Define the linear operator
        linOps = applyPreconditionedATA(AP, CP, sqrtLambP, M)

        x0 = np.zeros((AP.shape[1], 1))

        self.cg_count_reset()
        MTxP, info = cg(linOps, rhs, callback=self.cg_callback, tol=params["iterative"]["tolerance"], maxiter=params["iterative"]["max_iter"])
        MTxP = MTxP.reshape((-1, 1))

        print(f"cg_iter = {self.cg_count}")

        params["logger"].log({"iter": sef.cg_count})

        xP = spsolve_triangular(M.T, MTxP, lower=False)

        x = np.zeros_like(xP)
        x[P] = xP

        return x





