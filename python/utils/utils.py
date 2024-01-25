"""
utils.py: Utility functions that may be used by all files
Author: rogerhh
Date: 2023-05-30
"""

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
# from scikits.sparse.cholmod import cholesky, cholesky_AAt
from sksparse.cholmod import cholesky, cholesky_AAt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import cg, spsolve_triangular, inv
import scipy
import re

import gtsam

def chi2_red(nfg: gtsam.NonlinearFactorGraph, config: gtsam.Values, factor_dim) -> float:
    error = nfg.error(config)
    graph_dim = factor_dim
    dof = graph_dim - config.dim()
    assert(dof >= 0)
    if dof != 0:
        return 2 * error / dof
    else:
        return 0
        print("dof == 0!")
        raise RuntimeError

def readLCSteps(filename):
    lc_steps = []
    with open(filename, "r") as fin:
        while True:
            line = fin.readline()
            if not line:
                break

            if "SELECTED LOOP CLOSURE STEPS:" in line:
                line = fin.readline()
                num_steps = int(line.split()[1])
                line = fin.readline()
                arr = line.split()

                for i in range(num_steps):
                    lc_steps.append(int(arr[i]))

    return lc_steps

def getVarDim(theta, key):
    types = [(theta.atPose2, 3), (theta.atPoint2, 2), \
             (theta.atPose3, 6), (theta.atPoint3, 3)]
    flag = False

    for var_type, var_len in types:
        try:
            var_type(key)
            return var_len
        except RuntimeError:
            pass

    assert(flag)

def getVar(theta, key):
    types = [theta.atPose2, theta.atPoint2, \
             theta.atPose3, theta.atPoint3]
    flag = False

    for var_type in types:
        try:
            return var_type(key)
        except RuntimeError:
            pass

    assert(flag)

class CGcallback:
    count = 0

    def reset():
        CGcallback.count = 0

    def callback(xk):
        CGcallback.count = CGcallback.count + 1


def setUpIgoParams(params):
    # IgoSelectiveCholeskyUpdate2
    id_string = "selcholupdate2"
    params[id_string] = {}
    params[id_string]["percent_rows"] = 0.1

    # IgoSelectiveCholeskyUpdate3
    id_string = "selcholupdate3"
    params[id_string] = {}
    params[id_string]["percent_rows"] = 0.1

    # IgoSelectiveCholeskyUpdate4
    id_string = "selcholupdate4"
    params[id_string] = {}
    params[id_string]["percent_rows"] = 0.5

    # IgoBoundedDiff
    id_string = "boundeddiff"
    params[id_string] = {}
    params[id_string]["diff_threshold"] = 0.2

    # IgoBoundedDiff2
    id_string = "boundeddiff2"
    params[id_string] = {}
    params[id_string]["diff_threshold"] = 4
