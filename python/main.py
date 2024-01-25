import math

import matplotlib.pyplot as plt
import numpy as np

import yaml
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
from utils.problem_advancer3D import ProblemAdvancer3D
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
    parser.add_option("--preconditioner_type", "--pu_type", dest="pu_type",
                      default="baseline", help="What type of preconditioner updater to use.")
    parser.add_option("--params", dest="params_file",
                      default="/home/ubuntu/igo/python/params/params.yml", help="Path to params.yml")
    parser.add_option("--2D", dest="is_2D",
                      default=False, action="store_true", help="True if 2D dataset")
    parser.add_option("--3D", dest="is_3D",
                      default=False, action="store_true", help="True if 3D dataset")

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
    is_2D = option.is_2D
    is_3D = option.is_3D
    if is_3D:
        is_2D = False

    lc_steps_file = option.lc_steps_file
    lc_lookahead = int(option.lc_lookahead)

    pu_type = option.pu_type
    params_file = option.params_file

    # Command line input overwrites yaml file
    with open(params_file, "r") as params_fin:
        params = yaml.full_load(params_fin)
        params["mode"] = "pgo"
        if pu_type != "baseline":
            params["igo_id"] = pu_type
        if relin_threshold != 0.001:
            params["relin_threshold"] = relin_threshold
        if lc_lookahead != 1:
            params["lc_lookahead"] = lc_lookahead
        if lc_steps_file != "":
            params["lc_steps_file"] = lc_steps_file
        if params["dataset"] is not None:
            dataset = params["dataset"]
        if "is_2D" in params.keys() and params["is_2D"]:
            is_2D = True
            is_3D = False
        elif "is_3D" in params.keys() and params["is_3D"]:
            is_2D = False
            is_3D = True

    assert(is_2D ^ is_3D)

    params["output_yaml_obj"] = dict()

    lc_steps = readLCSteps(params["lc_steps_file"])
    params["lc_steps"] = lc_steps
    if "profile_lc_step_only" in params.keys() and params["profile_lc_step_only"]:
        lc_steps = [params["profile_lc_step"]]

    if is_2D:
        dataset_name = gtsam.findExampleDataFile(dataset)
        measurements = gtsam.NonlinearFactorGraph()
        # Prior on the first variable. Add it to factor graph for uniformity
        zero_prior = gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), \
                                            gtsam.noiseModel.Unit.Create(3))
        measurements.push_back(zero_prior)
        (dataset_measurements, initial) = gtsam.load2D(dataset_name)
        measurements.push_back(dataset_measurements)
    else:
        dataset_name = gtsam.findExampleDataFile(dataset)
        measurements = gtsam.NonlinearFactorGraph()
        # Prior on the first variable. Add it to factor graph for uniformity
        zero_prior = gtsam.PriorFactorPose3(0, gtsam.Pose3(), \
                                            gtsam.noiseModel.Unit.Create(6))
        measurements.push_back(zero_prior)
        (dataset_measurements, initial) = gtsam.load3D(dataset_name)
        measurements.push_back(dataset_measurements)
    print(f"dataset name: {dataset_name}")

    print(params)


    # params = {"igo_id": pu_type, "mode": "pgo", "relin_threshold": relin_threshold}
    # setUpIgoParams(params)

    state_estimation = StateEstimation(params)

    if is_2D:
        padv = ProblemAdvancer(measurements)
    else:
        padv = ProblemAdvancer3D(measurements)

    estimate = gtsam.Values()

    print(lc_steps)

    for lc_step in lc_steps:

        if lc_step > params["stop_step"]:
            break

        lookahead_step = lc_step - params["lc_lookahead"]
        params["step"] = lookahead_step

        # First set up the solution at lookahead_step
        # Get initial values and factors
        new_factors, new_theta, max_measurement_index = padv.advanceToStep(lookahead_step, estimate)
        estimate = state_estimation.setup_lc_step(new_theta, new_factors, params)

        print("Done setting up problem")

        for step in range(lookahead_step + 1, lc_step + 1):

            print(f"step = {step}")
            params["step"] = step

            if step == lc_step:
                print(f"Adding step {lc_step}")
                params["output_yaml_obj"][lc_step] = dict()

            # Get initial values and factors
            new_factors, new_theta, max_measurement_index = padv.advanceToStep(step, estimate)

            estimate = state_estimation.update(new_theta, new_factors, params)
            # print(f"Step = {step}\nEstimate = \n{estimate}")

        with open(params["output_file"], "w") as fout:
            fout.write(yaml.dump(params["output_yaml_obj"], default_flow_style=True))
        print(yaml.dump(params["output_yaml_obj"], default_flow_style=True))
        # input("Press key")

    pass
