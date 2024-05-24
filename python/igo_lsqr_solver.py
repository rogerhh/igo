from common_packages import *

from utils.linear_operator import applyPreconditionedATA
from igo_iterative_solver_base import IgoIterativeSolverBase

class IgoLsqrSolver(IgoIterativeSolverBase):
    """
    Solves the linear least squares problem using the LSQR method
    We are solving problem min ||[A \\ iC \\ sqrtLamb]x - [b \\ id \\ 0]||^2
    """

    id_string = "lsqr"

    def __init__(self, params):
        super().__init__(params)
        self.cg_count = 0

    def iter_count_reset(self):
        self.iter_count = 0

    def iter_callback(self, x):
        self.iter_count += 1

    def solve(self, A, b, C, d, sqrtLamb, M, P, params):
        # Permute columns of A, C, sqrtLamb with P
        AP = A[:, P]
        CP = C[:, P]
        sqrtLambP = sqrtLamb[:, P]

        # Set up rhs = [b \\ id \\ 0]
        A_rows = AP.shape[0]
        C_rows = CP.shape[0]
        sqrtLamb_rows = sqrtLambP.shape[0]
        rhs = np.zeros((A_rows + C_rows + sqrtLamb_rows, 1))
        rhs[:A_rows] = b.A
        rhs[A_rows:A_rows + C_rows] = d.A
        rhs[A_rows + C_rows:] = 0

        # Define the linear operator
        linOps = applyPreconditionedA(AP, CP, sqrtLambP, M)

        x0 = np.zeros((AP.shape[1], 1))

        print(linOps.shape, rhs.shape, A_rows, C_rows, sqrtLamb_rows)

        MTxP, istop, itn, normr = scipy.sparse.linalg.lsqr(linOps, rhs.squeeze(), atol=float(params["iterative"]["tolerance"]), btol=float(params["iterative"]["tolerance"]), iter_lim=int(params["iterative"]["max_iter"]))[:4]

        print(f"istop = {istop}, itn = {itn}, normr = {normr}")
        params["logger"].log({"iter": itn})

        xP = spsolve_triangular(M.T, MTxP, lower=False)

        x = np.zeros_like(xP)
        x[P] = xP

        return x
        





