import math

import matplotlib.pyplot as plt
import numpy as np

import gtsam
import gtsam.utils.plot as gtsam_plot

from typing import List

from optparse import OptionParser

from scikits.sparse.cholmod import cholesky, cholesky_AAt
# from sksparse.cholmod import cholesky, cholesky_AAt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import cg, spsolve_triangular, inv
import matplotlib.pyplot as plt
import scipy

from utils.problem_advancer import ProblemAdvancer
from utils.utils import *

from nonlinear_opt import StateEstimation

if __name__ == "__main__":
    # Driver code for a generic 2D dataset
    # 1. For any given datasets and flags, read in the dataset
    # 2. For each step, we have 2 options,
    #   1) Fast forward to particular step
    #       a) Gather factors from this step to the new step
    #       b) Run StateEstimation.update_* with the new factors and correct theta
    #   2) Take one step 
    #       a) Get the new theta and new factors at that step. Append new theta to theta
    #       b) Run StateEstimation.update_* with the new factors and incorrect theta

    parser = OptionParser()
    parser.add_option("-f", "--dataset", dest="dataset",
                      default="", help="Name of dataset")
    parser.add_option("-k", "--K", dest="K",
                      default="1", help="Period of ISAM2 updates")
    parser.add_option("-e", "--epsilon", dest="epsilon",
                      default="0.01", help="Error tolerance")
    parser.add_option("-m", "--max_iter", dest="max_iter",
                      default="10", help="Number of inner loop iterations")
    parser.add_option("-d", "--d_error", dest="d_error",
                      default="0.001", help="If error does not reduce more than d_error, \
                                             we consider it converged")
    parser.add_option("--relinearize_skip", dest="relinearize_skip",
                      default="1", help="Number of steps between relinearization of variable")
    parser.add_option("--print_frequency", dest="print_frequency",
                      default="100", help="Frequency of printing")
    parser.add_option("--num_steps", dest="num_steps",
                      default="100000000", help="Maximum steps")
    parser.add_option("--relin_threshold", dest="relin_threshold",
                      default="0.001", help="Delta norm to relinearize variable")
    parser.add_option("--lc_steps_file", dest="lc_steps_file",
                      default="", help="File listing all the loop closure steps.")
    parser.add_option("--lc_lookahead", dest="lc_lookahead",
                      default="1", help="How many steps before the LC step to run factorization.")
    parser.add_option("--preconditioner_type", dest="pu_type",
                      default="identity", help="What type of preconditioner updater to use.")

    (option, args) = parser.parse_args()
    dataset = option.dataset
    K = int(option.K)
    relinearize_skip = int(option.relinearize_skip)
    epsilon = float(option.epsilon)
    d_error = float(option.d_error)
    max_iter = int(option.max_iter)
    print_frequency = int(option.print_frequency)
    num_steps = int(option.num_steps)
    relin_threshold = float(option.relin_threshold)

    lc_steps_file = option.lc_steps_file
    lc_steps = readLCSteps(lc_steps_file)
    lc_lookahead = int(option.lc_lookahead)

    dataset_name = gtsam.findExampleDataFile(dataset)
    measurements = gtsam.NonlinearFactorGraph()
    # Prior on the first variable. Add it to factor graph for uniformity
    zero_prior = gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), \
                                        gtsam.noiseModel.Unit.Create(3))
    measurements.push_back(zero_prior)
    (dataset_measurements, initial) = gtsam.load2D(dataset_name)
    measurements.push_back(dataset_measurements)

    params = {"igo_id": "selcholupdate2", "mode": "pgo", "relin_threshold": 1e-3}
    setUpIgoParams(params)

    state_estimation = StateEstimation(params)

    padv = ProblemAdvancer(measurements)

    estimate = gtsam.Values()

    for lc_step in lc_steps:

        if lc_step > 5000:
            break

        lookahead_step = lc_step - lc_lookahead

        # First set up the solution at lookahead_step
        # Get initial values and factors
        new_factors, new_theta, max_measurement_index = padv.advanceToStep(lookahead_step, estimate)
        setup_params = deepcopy(params)
        setup_params["relin_threshold"] = 0
        setup_params["selcholupdate2"]["percent_rows"] = 1
        estimate = state_estimation.update(new_theta, new_factors, setup_params)

        print("Done setting up problem")

        for step in range(lookahead_step + 1, lc_step + 1):

            print(f"step = {step}")

            # Get initial values and factors
            new_factors, new_theta, max_measurement_index = padv.advanceToStep(step, estimate)

            estimate = state_estimation.update(new_theta, new_factors, params)
            # print(f"Step = {step}\nEstimate = \n{estimate}")

        input("Press key")

    pass
