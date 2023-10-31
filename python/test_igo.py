from igo import *
import math

import matplotlib.pyplot as plt
import numpy as np

from typing import List

from optparse import OptionParser

from scikits.sparse.cholmod import cholesky, cholesky_AAt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import cg, spsolve_triangular, inv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    igo = IgoBaseline()

    A_tilde = csr_matrix(([], ([], [])), shape=(0, 0))
    b_tilde = csr_matrix(([], ([], [])), shape=(0, 1))
    A_hat = csr_matrix(([1, 1, 1], ([0, 1, 2], [0, 1, 2])))
    b_hat = csr_matrix(([3, 1, 2], ([0, 1, 2], [0, 0, 0])))

    diagLamb = csr_matrix(([1e-12, 1e-12, 1e-12], ([0, 1, 2], [0, 1, 2])))

    x = igo.incremental_opt(A_tilde, b_tilde, A_hat, b_hat, diagLamb)

    print(x)


