from common_packages import *

class IgoSolverBase(ABC):
    """
    This is wrapper class around an iterative solver.
    Given inputs A, b, C, d, M, P, return the solution of (AAt - CCt)x = Ab - Cd
    Using M as a preconditioner
    """

    def __init__(self, params):
        pass

    @abstractmethod
    def solve(self, A, b, C, d, M, P, params):
        raise NotImplementedError


    # Common interface to construct the linear operator and to permute/unpermute A, C to P
    def set_up_inputs(self, A, b, C, d, M, P):
        pass

    def set_up_outputs(self, x):
        pass

class IgoPcgne(IgoSolverBase):
    def __init__(self, params):
        super.__init__(params)

    def solve(self, A, b, C, d, M, P, params):
        pass

class IgoLsqr(IgoSolverBase):
    def __init__(self, params):
        super.__init__(params)

    def solve(self, A, b, C, d, M, P, params):
        pass
