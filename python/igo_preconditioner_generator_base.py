from common_packages import *

class IgoPreconditionerGeneratorBase(ABC):
    """
    Generate the preconditioner for the iterative solver
    """
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def generate_preconditioner(self, A_tilde, A_hat, C_tilde, C_hat, sqrtLamb_tilde, sqrtLamb_hat):
        raise NotImplementedError("generate_preconditioner not implemented")

