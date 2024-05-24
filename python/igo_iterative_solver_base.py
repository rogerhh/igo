from common_packages import *

class IgoIterativeSolverBase(ABC):
    """
    Specifies the iterative solver used in IgoIterative
    """

    def __init__(self, params):
        self.params = params

    # Solves the least-squares problem min ||[A \\ iC \\ sqrtLamb]x - [b \\ id \\ 0]||^2
    # by solving the normal equations of the form (A^T A - C^T C + sqrtLamb^T sqrtLamb)x = A^T b - C^T d
    # A^T A - C^T C + sqrtLamb^T sqrtLamb is symmetric positive definite
    # M is the lower triangular preconditioner and P is the permutation vector
    @abstractmethod
    def solve(self, A, b, C, d, sqrtLamb, M, P, params):
        raise NotImplementedError("solve not implemented")
    
