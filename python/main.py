"""
Driving code for the Python proof-of-concept implementation of igo. 
The Python code is written for interface testing, debugging,
and playing around with preconditioners
"""

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
    # Driver code for a generic 2D/3D dataset
    # 1. For any given datasets and flags, read in the dataset
    # 2. For each step, we have 2 options,
    #   1) Fast forward to particular step
    #       a) Gather factors from this step to the new step
    #       b) Run StateEstimation.update_* with the new factors and correct theta
    #   2) Take one step 
    #       a) Get the new theta and new factors at that step. Append new theta to theta
    #       b) Run StateEstimation.update_* with the new factors and incorrect theta

    parser = OptionParser()
    parser.add_option("--params", dest="params_file",
                      default=None, help="Path to params.yml")

    (option, args) = parser.parse_args()

    params_file = option.params_file

    # Command line input overwrites yaml file
    with open(params_file, "r") as params_fin:
        params = yaml.full_load(params_fin)

    params["output_yaml_obj"] = dict()

    lc_steps = readLCSteps(params["lc_steps_file"])
    params["lc_steps"] = lc_steps
    if "profile_lc_step_only" in params.keys() and params["profile_lc_step_only"]:
        lc_steps = [params["profile_lc_step"]]

    is_3D = params["is_3D"]
    is_2D = not is_3D
    dataset = params["dataset"]

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
