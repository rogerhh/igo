class IgoIncompleteCholeskyStrongConnection(IgoBase):
    """
    SelectiveCholeskyUpdate picks the high norm rows to update Cholesky with and use 
    as a preconditioner in PCG
    """
    id_string = "incompletecholstrongconnection"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None

    """
    This function is defined a bit differently than in the paper. Here, A_tilde directly replaces 
    coressponding entries in A
    Implements the following: 
        A[nonzero(A_tilde)] = A_tilde
        A = [A; A_hat]
        b[nonzero(b_tilde)] = b_tilde
        b = [b; b_hat]
        return solve(argmin_x \|Ax - b\|_2^2)
    """
    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):

        return self.x

    def marginalize(self, cols):
        pass

class IgoIncompleteCholeskyValueFilter(Igo):
    """
    SelectiveCholeskyUpdate picks the high norm rows to update Cholesky with and use 
    as a preconditioner in PCG
    """
    id_string = "incompletecholvaluefilter"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None

    """
    This function is defined a bit differently than in the paper. Here, A_tilde directly replaces 
    coressponding entries in A
    Implements the following: 
        A[nonzero(A_tilde)] = A_tilde
        A = [A; A_hat]
        b[nonzero(b_tilde)] = b_tilde
        b = [b; b_hat]
        return solve(argmin_x \|Ax - b\|_2^2)
    """
    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):

        return self.x

    def marginalize(self, cols):
        pass

