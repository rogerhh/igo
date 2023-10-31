"""
sparse_linear_system.py: We have a collection of Factor's, where each Factor owns a set of continuous indices. Linearizing/Relinearizing a Factor involves changing the entries of A_data, b_data at the indices that it owns A_data[indices], b_data[indices]. The row and column indices are denoted by A_rows[indices] and A_cols[indices]
A sparse_linear_system owns a list of Factor's, (A_data, A_rows, A_cols) which are 3 vectors that represent a sparse matrix A, and a dense vector b. b can be represented by only 1 vector because all indices are filled and there are no duplicated indices
"""

from scikits.sparse.cholmod import cholesky, cholesky_AAt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import cg, spsolve_triangular, inv
import matplotlib.pyplot as plt
import scipy
import numpy as np

from utils.utils import *

import gtsam

class Factor:
    def __init__(self, factor, system):
        self.indices = []
        # A reference to the system so we can change the matrix entries
        self.system = system
        pass

    # Update the entries at the indices that we own
    def linearize(self, theta):
        pass

class SparseLinearSystem:

    def __init__(self):
        self.factors = []
        self.key_to_col = [0]
        self.key_to_factors = []
        self.factor_to_row = [0]

        # Keeps track of the linearization point
        self.theta = gtsam.Values()
        self.delta = gtsam.VectorValues()

    def addVariables(self, new_theta):
        self.theta.insert(new_theta)

        # Only need to add dimension of new theta here
        for key in new_theta.keys():
            var_len = getVarDim(new_theta, key)

            self.key_to_col.append(self.key_to_col[-1] + var_len)
            self.key_to_factors.append([])

    def addFactor(self, factor):
        factor_index = len(self.factors)
        self.factors.append(factor)
        self.factor_to_row.append(self.factor_to_row[-1] + factor.dim())

        for key in factor.keys():
            self.key_to_factors[key].append(factor_index)

    def addFactors(self, new_factors):
        for i in range(new_factors.size()):
            self.addFactor(new_factors.at(i))

    def getLinearFactorCSR(self, factor_index, A_rows, A_cols, A_data, b_rows, b_cols, b_data):
        factor = self.factors[factor_index]
        jacobian_factor = factor.linearize(self.theta)
        matrixA = jacobian_factor.getA()
        vectorb = jacobian_factor.getb()

        factor_start_row = self.factor_to_row[factor_index]

        col_start = 0
        for key in factor.keys():
            key_start_col = self.key_to_col[key]
            width = self.key_to_col[key + 1] - key_start_col

            for j in range(width):
                c = key_start_col + j
                for i in range(factor.dim()):
                    r = factor_start_row + i
                    A_rows.append(r)
                    A_cols.append(c)
                    A_data.append(matrixA[i, col_start + j])

            col_start += width

        for i in range(factor.dim()):
            b_rows.append(factor_start_row + i)
            b_cols.append(0)
            b_data.append(vectorb[i])

        
    def linearizeFactor(self, factor_index, theta):
        factor = self.factors[factor_index]
        jacobian_factor = factor.linearize(theta)
        matrixA = jacobian_factor.getA()
        vectorb = jacobian_factor.getb()

        A_index = self.A_start_indices[factor_index]
        A_start_row = self.b_start_indices[factor_index]
        col_start = 0

        for key in factor.keys():
            (col, width) = self.key_to_col[key]

            for j in range(width):
                c = col + j
                for i in range(factor.dim()):
                    r = A_start_row + i

                    assert(self.A_rows[A_index] == r)
                    assert(self.A_cols[A_index] == c)

                    self.A_data[A_index] = matrixA[i, col_start + j]

                    # 2023/09/29: Add this to prevent csr_matrix from removing nonzero entry
                    if self.A_data[A_index] == 0:
                        self.A_data[A_index] = 1e-12

                    A_index += 1

            col_start += width

        b_index = self.b_start_indices[factor_index]
        for i in range(factor.dim()):
            self.b_data[b_index + i] = vectorb[i]

    def linearizeAll(self, theta):
        for factor_index in range(len(self.factors)):
            self.linearizeFactor(factor_index, theta)

    def linearizeNew(self, index_start, index_end, theta):
        for factor_index in range(index_start, index_end):
            self.linearizeFactor(factor_index, theta)

    def linearizeSet(self, relin_factors, theta):
        for factor_index in relin_factors:
            self.linearizeFactor(factor_index, theta)


    # Return A matrix and b vector obtained at step and with linearization point theta
    # Adding all the measurements until step
    def getSystem(self):
        A = csr_matrix((self.A_data, (self.A_rows, self.A_cols)))
        b_rows = range(len(self.b_data))
        b_cols = [0 for _ in b_rows]
        b = csr_matrix((self.b_data, (b_rows, b_cols)))
        # b = np.array(self.b_data).reshape((-1, 1))
        return A, b
