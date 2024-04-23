from common_packages import 

class IgoIdentity(IgoPreconditionerGeneratorBase):
    """
    Identity preconditioner
    """
    id_string = "identity"

    def __init__(self, params):
        super().__init__(params)

    def generate_preconditioner(self, A_tilde, A_hat, C_tilde, C_hat, sqrtLamb_tilde, sqrtLamb_hat):
        return np.eye(A_hat.shape[1])

