from common_packages import *

from igo_preconditioner_generator_base import IgoPreconditionerGeneratorBase

class IgoIdentity(IgoPreconditionerGeneratorBase):
    """
    Identity preconditioner
    """
    id_string = "identity"

    def __init__(self, params):
        super().__init__(params)

    def generate_preconditioner(self, A, A_tilde, A_hat, C, C_tilde, C_hat, sqrtLamb, sqrtLamb_tilde, sqrtLamb_hat, params):
        return np.eye(A_hat.shape[1]), np.arange(A_hat.shape[1])

