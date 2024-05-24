from common_packages import *

"""
Base class that all igo implementations should derive from. 
Implements incremental_opt() and marginalize()
"""
class IgoBase(ABC):

    # Initialize the class with parameters A, b, C, d, sqrtLamb
    

    def __init__(self, params):
        self.params = params

        self.A = csr_matrix(([], ([], [])), shape=(0, 0))
        self.b = csr_matrix(([], ([], [])), shape=(0, 0))
        self.C = csr_matrix(([], ([], [])), shape=(0, 0))
        self.d = csr_matrix(([], ([], [])), shape=(0, 0))

        self.sqrtLamb = csr_matrix(([], ([], [])), shape=(0, 0))

    @abstractmethod
    def incremental_opt(self, 
                        old_size, new_size,
                        A_tilde, b_tilde, 
                        A_hat, b_hat, 
                        C_tilde, d_tilde,
                        C_hat, d_hat,
                        sqrtLamb_tilde,
                        sqrtLamb_hat, 
                        params):
        raise NotImplementedError

    @abstractmethod
    def marginalize(self, keys):
        raise NotImplementedError

