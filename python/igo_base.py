from common_packages import *

"""
Base class that all igo implementations should derive from. 
Implements incremental_opt() and marginalize()
"""
class IgoBase(ABC):

    # Initialize the class with parameters A, b, C, d, sqrtLamb
    

    def __init__(self, params):
        self.A = csr_matrix(([], ([], [])), shape=(0, 0))
        self.b = csr_matrix(([], ([], [])), shape=(0, 0))
        self.negA = csr_matrix(([], ([], [])), shape=(0, 1))
        self.negb = csr_matrix(([], ([], [])), shape=(0, 1))

        self.diagLamb = csr_matrix(([], ([], [])), shape=(0, 0))

    @abstractmethod
    def setup_lc_step(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        raise NotImplementedError

    @abstractmethod
    def incremental_opt(self, A_tilde, b_tilde, A_hat, b_hat, diagLamb, params):
        raise NotImplementedError

    @abstractmethod
    def marginalize(self, keys):
        raise NotImplementedError

