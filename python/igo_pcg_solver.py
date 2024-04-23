from common import *

from igo_iterative_solver_base import IgoIterativeSolverBase

class IgoPcgSolver(IgoIterativeSolverBase):
    """
    Solves the linear system using the preconditioned conjugate gradient method
    """

    id_string = "pcg"

    def __init__(self, params):
        super().__init__(params)

    def solve(self, A, b, C, d, sqrtLamb, M, P):
        """
        Solve the linear system using the preconditioned conjugate gradient method
        """
        # The preconditioned conjugate gradient method solves the linear system Ax = b
        # where A is a symmetric positive definite matrix
        # The preconditioner is a matrix M that approximates the inverse of A
        # The method is based on the conjugate gradient method
        # The method is iterative and requires the matrix A to be symmetric positive definite
        # The method is based on the conjugate gradient method
