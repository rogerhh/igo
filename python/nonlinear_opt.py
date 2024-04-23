"""
Base class for some nonlinear state estimation algorithms
Each algorithms responsibility is to supply the incremental linear solver 
relinearized parts of the matrix at each iteration
"""

import gtsam
from abc import ABC, abstractmethod
from copy import deepcopy
from scipy.sparse import csc_matrix, csr_matrix, eye
from utils.sparse_linear_system import SparseLinearSystem
from utils.logger import NoLogger, log
from utils.utils import *
from igo import *

class StateEstimation(ABC):
    def __init__(self, params):
        self.default_relin_threshold = params["relin_threshold"]
        self.spls = SparseLinearSystem()
        self.diagLamb = csr_matrix(([], ([], [])), shape=(0, 0))
        self.estimate = gtsam.Values()
        self.nfg = gtsam.NonlinearFactorGraph()

        self.igo = None
        for igo_type in igo_types:
            if params["igo_id"] == igo_type.id_string:
                self.igo = igo_type(params)
        assert(self.igo is not None), "Unknown Igo ID"

    # Map factors to rows and variables to cols
    def add_new_vars_factors(self, new_theta, new_factors):
        self.spls.addVariables(new_theta)
        self.spls.addFactors(new_factors)
        self.nfg.push_back(new_factors)

    # Run nonlinear optimization until convergence
    def setup_lc_step(self, new_theta, new_factors, params):
        setup_params = deepcopy(params)
        setup_params["relin_threshold"] = 0
        setup_params["setup_lc_step"] = True
        setup_params["default_num_outer_iter"] = 10

        with NoLogger():
            return self.update(new_theta, new_factors, setup_params)

    def update(self, new_theta, new_factors, params):
        if not "relin_threshold" in params.keys():
            exit(0)
            params["relin_threshold"] = self.default_relin_threshold

        update_count = 0
        params["outer_iter"] = update_count
        updated = self.update_impl(new_theta=new_theta, new_factors=new_factors, params=params)
        chi2 = chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1])
        log(f"chi2 = {chi2}\n\n")
        while updated and update_count < params["default_num_outer_iter"]:
            update_count += 1
            params["outer_iter"] = update_count
            print(f"update_count = {update_count}, chi2 = {chi2}")
            updated = self.update_impl(new_theta=gtsam.Values(), new_factors=gtsam.NonlinearFactorGraph(), params=params)
            chi2 = chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1])
            log(f"chi2 = {chi2}\n\n")

        return self.estimate

    # Returns True if an update is made, False otherwise
    def update_impl(self, new_theta, new_factors, params):

        # 1. Pick algorithm-specific factors to relinearize and form [A_tilde b_tilde]
        # 2. Add new variables and factors in
        # 2. Relinearize new_factors and form [A_hat b_hat]
        # 3. Pick appropriate diagonal lambda value
        # 4. Call incremental_opt to get delta
        # 5. Assign delta to algorithm specific theta entries

        if params["mode"] == "vio":
            relin_keys, relin_factors, theta_update, delta_update = self.vio_get_relin_factors(params)
        elif params["mode"] == "fullba":
            relin_keys, relin_factors, theta_update, delta_update = self.fullba_get_relin_factors(params)
        elif params["mode"] == "localba":
            relin_keys, relin_factors, theta_update, delta_update \
                    = self.localba_get_relin_factors(params, new_theta, new_factors)
        elif params["mode"] == "pgo":
            relin_keys, relin_factors, theta_update, delta_update = self.pgo_get_relin_factors(params)
        else:
            raise NotImplementedError

        # All keys whose delta entries are greater than relin threshold will have delta entries
        # set to 0. Corresponding entries in theta will be updated
        self.spls.delta.update(delta_update)
        self.spls.theta = self.spls.theta.retract(theta_update)
        # print("after delta update\n", self.spls.delta)
        # print("after theta update\n", self.spls.theta)

        Atilde_rows = []
        Atilde_cols = []
        Atilde_data = []
        btilde_rows = []
        btilde_cols = []
        btilde_data = []

        for factor_index in relin_factors:
            self.spls.getLinearFactorCSR(factor_index, \
                                         A_rows=Atilde_rows, A_cols=Atilde_cols, \
                                         A_data=Atilde_data, b_rows=btilde_rows, \
                                         b_cols=btilde_cols, b_data=btilde_data)

        old_num_factors = len(self.spls.factors)
        self.add_new_vars_factors(new_theta, new_factors)
        new_num_factors = len(self.spls.factors)

        Ahat_rows = []
        Ahat_cols = []
        Ahat_data = []
        bhat_rows = []
        bhat_cols = []
        bhat_data = []

        for i in range(old_num_factors, new_num_factors):
            self.spls.getLinearFactorCSR(i, \
                                         A_rows=Ahat_rows, A_cols=Ahat_cols, A_data=Ahat_data, \
                                         b_rows=bhat_rows, b_cols=bhat_cols, b_data=bhat_data)

        if len(Atilde_data) == 0:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)), shape=self.igo.A.shape)
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)), shape=self.igo.b.shape)
        else:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)), shape=self.igo.A.shape)
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)), shape=self.igo.b.shape)
        if len(Ahat_data) == 0:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)), shape=self.igo.A.shape)
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)), shape=self.igo.b.shape)
        else:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)))
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)))

        if len(Atilde_data) + len(Ahat_data) == 0:
            return False

        # Set C and d to 0 for now. Set sqrtLamb to 0 for now
        C_tilde = csr_matrix(([], ([], [])), shape=(0, A_tilde.shape[1]))
        d_tilde = csr_matrix(([], ([], [])), shape=(A_tilde.shape[1], 1))
        C_hat = csr_matrix(([], ([], [])), shape=(0, A_hat.shape[1]))
        d_hat = csr_matrix(([], ([], [])), shape=(A_hat.shape[1], 1))
        sqrtLamb_tilde = csr_matrix(([], ([], [])), shape=(A_tilde.shape[1], A_tilde.shape[1]))
        sqrtLamb_hat = csr_matrix(([], ([], [])), shape=(A_hat.shape[1], A_hat.shape[1]))

        delta_vec = self.igo.incremental_opt(A_tilde=A_tilde, \
                                             b_tilde=b_tilde, \
                                             A_hat=A_hat, \
                                             b_hat=b_hat, \
                                             C_tilde=C_tilde, \
                                             d_tilde=d_tilde, \
                                             C_hat=C_hat, \
                                             d_hat=d_hat, \
                                             sqrtLamb_tilde=sqrtLamb_tilde, \
                                             sqrtLamb_hat=sqrtLamb_hat, \
                                             params=params)

        if isinstance(delta_vec, np.ndarray):
            pass
        elif isinstance(delta_vec, csc_matrix):
            delta_vec = delta_vec.A

        # Check linearization and update vector values
        delta = gtsam.VectorValues()
        for key in range(len(self.spls.key_to_col) - 1):
            col = self.spls.key_to_col[key]
            width = self.spls.key_to_col[key + 1] - col
            delta_key = delta_vec[col:col+width]
            delta.insert(key, delta_key)

        self.spls.delta = delta

        self.estimate = self.spls.theta.retract(self.spls.delta)

        return True

    def vio_get_relin_factors(self, params):
        # Check delta for which variables need to be relinearized
        # Only check keys that are not too far back in time
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()

        vio_horizon = 5
        max_key = self.spls.delta.size() - 1
        start_key = max_key - vio_horizon if max_key >= vio_horizon else 0
        for key in range(self.spls.delta.size(), self.spls.delta.size()):
            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm >= params["relin_threshold"]:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        return relin_keys, relin_factors, theta_update, delta_update

    def fullba_get_relin_factors(self, params):
        # Check delta for which variables need to be relinearized
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()
        for key in range(self.spls.delta.size()):
            var = getVar(self.spls.theta, key)
            if not (isinstance(var, gtsam.Pose2) or isinstance(var, gtsam.Pose3)):
                continue

            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm >= params["relin_threshold"]:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        return relin_keys, relin_factors, theta_update, delta_update

    def localba_get_relin_factors(self, params, new_theta, new_factors):
        # Check delta for which variables need to be relinearized
        # Add all keys that are two factors away
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()

        if self.spls.delta.size() == 0:
            return relin_keys, relin_factors, theta_update, delta_update

        max_old_key = self.spls.delta.size()
        max_new_key = max_old_key + new_theta.size()

        keys_to_check = set()
        checked_keys = set()

        if new_factors.size() > 0:
            for factor_index in range(new_factors.size()):
                factor = new_factors.at(factor_index)
                for factor_key in factor.keys():
                    if factor_key < max_old_key:
                        keys_to_check.add(factor_key)
                        checked_keys.add(factor_key)
        else:
            last_key = max_old_key - 1
            for factor_index in self.spls.key_to_factors[last_key]:
                factor = self.spls.factors[factor_index]
                for factor_key in factor.keys():
                    keys_to_check.add(factor_key)
                    checked_keys.add(factor_key)

        for key in keys_to_check:
            for factor_index in self.spls.key_to_factors[key]:
                factor = self.spls.factors[factor_index]
                for factor_key in factor.keys():
                    assert(factor_key < max_old_key)
                    checked_keys.add(factor_key)


        for key in checked_keys:

            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm >= params["relin_threshold"]:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        print(relin_keys)
        # print("in localba relin factors\n", delta_update)
        # print(theta_update)

        return relin_keys, relin_factors, theta_update, delta_update

    def pgo_get_relin_factors(self, params):
        # Check delta for which variables need to be relinearized
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()
        for key in range(self.spls.delta.size()):
            var = getVar(self.spls.theta, key)
            if not (isinstance(var, gtsam.Pose2) or isinstance(var, gtsam.Pose3)):
                continue

            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm > params["relin_threshold"]:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        return relin_keys, relin_factors, theta_update, delta_update

    def update_vio(self, new_theta, new_factors, relin_threshold=None):
        if relin_threshold is None:
            relin_threshold = self.default_relin_threshold

        self.update_vio_impl(new_theta, new_factors, relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))
        self.update_vio_impl(gtsam.Values(), gtsam.NonlinearFactorGraph(), relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))

        return self.estimate

    def update_vio_impl(self, new_theta, new_factors, relin_threshold):

        # 1. Pick algorithm-specific factors to relinearize and form [A_tilde b_tilde]
        # 2. Add new variables and factors in
        # 2. Relinearize new_factors and form [A_hat b_hat]
        # 3. Pick appropriate diagonal lambda value
        # 4. Call incremental_opt to get delta
        # 5. Assign delta to algorithm specific theta entries

        # Check delta for which variables need to be relinearized
        # Only check keys that are not too far back in time
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()

        vio_horizon = 5
        max_key = self.spls.delta.size() - 1
        start_key = max_key - vio_horizon if max_key >= vio_horizon else 0
        for key in range(self.spls.delta.size(), self.spls.delta.size()):
            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm >= relin_threshold:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        # All keys whose delta entries are greater than relin threshold will have delta entries
        # set to 0. Corresponding entries in theta will be updated
        self.spls.delta.update(theta_update)
        self.spls.theta = self.spls.theta.retract(theta_update)

        Atilde_rows = []
        Atilde_cols = []
        Atilde_data = []
        btilde_rows = []
        btilde_cols = []
        btilde_data = []

        for factor_index in relin_factors:
            self.spls.getLinearFactorCSR(factor_index, \
                                         A_rows=Atilde_rows, A_cols=Atilde_cols, \
                                         A_data=Atilde_data, b_rows=btilde_rows, \
                                         b_cols=btilde_cols, b_data=btilde_data)

        old_num_factors = len(self.spls.factors)
        self.add_new_vars_factors(new_theta, new_factors)
        new_num_factors = len(self.spls.factors)

        Ahat_rows = []
        Ahat_cols = []
        Ahat_data = []
        bhat_rows = []
        bhat_cols = []
        bhat_data = []

        for i in range(old_num_factors, new_num_factors):
            self.spls.getLinearFactorCSR(i, \
                                         A_rows=Ahat_rows, A_cols=Ahat_cols, A_data=Ahat_data, \
                                         b_rows=bhat_rows, b_cols=bhat_cols, b_data=bhat_data)

        if len(Atilde_data) == 0:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)), shape=(0, 0))
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)), shape=(0, 0))
        else:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)))
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)))
        if len(Ahat_data) == 0:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)), shape=(0, 0))
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)), shape=(0, 0))
        else:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)))
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)))

        delta_vec = self.igo.incremental_opt(A_tilde=A_tilde, b_tilde=b_tilde, \
                                             A_hat=A_hat, b_hat=b_hat, diagLamb=self.diagLamb)

        delta_vec = delta_vec.A

        # Check linearization and update vector values
        delta = gtsam.VectorValues()
        for key in range(len(self.spls.key_to_col) - 1):
            col = self.spls.key_to_col[key]
            width = self.spls.key_to_col[key + 1] - col
            delta_key = delta_vec[col:col+width]
            delta.insert(key, delta_key)

        self.spls.delta = delta

        self.estimate = self.spls.theta.retract(self.spls.delta)

    def update_localba(self, delta_vec, new_theta, new_factors, relin_threshold=None):
        pass

    def update_fullba(self, delta_vec, new_theta, new_factors, relin_threshold=None):
        if relin_threshold is None:
            relin_threshold = self.default_relin_threshold

        self.update_fullba_impl(new_theta, new_factors, relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))
        self.update_fullba_impl(gtsam.Values(), gtsam.NonlinearFactorGraph(), relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))

        return self.estimate


    def update_pgo(self, new_theta, new_factors, relin_threshold=None):
        if relin_threshold is None:
            relin_threshold = self.default_relin_threshold

        self.update_pgo_impl(new_theta, new_factors, relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))
        self.update_pgo_impl(gtsam.Values(), gtsam.NonlinearFactorGraph(), relin_threshold)
        print("chi2 = ", chi2_red(self.nfg, self.estimate, self.spls.factor_to_row[-1]))

        return self.estimate

    def update_pgo_impl(self, new_theta, new_factors, relin_threshold):

        # 1. Pick algorithm-specific factors to relinearize and form [A_tilde b_tilde]
        # 2. Add new variables and factors in
        # 2. Relinearize new_factors and form [A_hat b_hat]
        # 3. Pick appropriate diagonal lambda value
        # 4. Call incremental_opt to get delta
        # 5. Assign delta to algorithm specific theta entries

        # Check delta for which variables need to be relinearized
        relin_factors = set()
        relin_keys = set()
        theta_update = gtsam.VectorValues()
        delta_update = gtsam.VectorValues()
        for key in range(self.spls.delta.size()):
            var = getVar(self.spls.theta, key)
            if not (isinstance(var, gtsam.Pose2) or isinstance(var, gtsam.Pose3)):
                continue

            delta_key = self.spls.delta.at(key)
            delta_norm = np.linalg.norm(delta_key, ord=np.inf)
            if delta_norm >= relin_threshold:
                # This delta will be retracted from theta
                theta_update.insert(key, delta_key)
                delta_update.insert(key, np.zeros_like(delta_key))
                relin_factors = relin_factors | set(self.spls.key_to_factors[key])
                relin_keys.add(key)

            else:
                # This delta will not be retracted from theta
                delta_update.insert(key, delta_key)

        # All keys whose delta entries are greater than relin threshold will have delta entries
        # set to 0. Corresponding entries in theta will be updated
        self.spls.delta.update(theta_update)
        self.spls.theta = self.spls.theta.retract(theta_update)

        Atilde_rows = []
        Atilde_cols = []
        Atilde_data = []
        btilde_rows = []
        btilde_cols = []
        btilde_data = []

        for factor_index in relin_factors:
            self.spls.getLinearFactorCSR(factor_index, \
                                         A_rows=Atilde_rows, A_cols=Atilde_cols, \
                                         A_data=Atilde_data, b_rows=btilde_rows, \
                                         b_cols=btilde_cols, b_data=btilde_data)

        old_num_factors = len(self.spls.factors)
        self.add_new_vars_factors(new_theta, new_factors)
        new_num_factors = len(self.spls.factors)

        Ahat_rows = []
        Ahat_cols = []
        Ahat_data = []
        bhat_rows = []
        bhat_cols = []
        bhat_data = []

        for i in range(old_num_factors, new_num_factors):
            self.spls.getLinearFactorCSR(i, \
                                         A_rows=Ahat_rows, A_cols=Ahat_cols, A_data=Ahat_data, \
                                         b_rows=bhat_rows, b_cols=bhat_cols, b_data=bhat_data)

        if len(Atilde_data) == 0:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)), shape=(0, 0))
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)), shape=(0, 0))
        else:
            A_tilde = csr_matrix((Atilde_data, (Atilde_rows, Atilde_cols)))
            b_tilde = csr_matrix((btilde_data, (btilde_rows, btilde_cols)))
        if len(Ahat_data) == 0:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)), shape=(0, 0))
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)), shape=(0, 0))
        else:
            A_hat = csr_matrix((Ahat_data, (Ahat_rows, Ahat_cols)))
            b_hat = csr_matrix((bhat_data, (bhat_rows, bhat_cols)))

        delta_vec = self.igo.incremental_opt(A_tilde=A_tilde, b_tilde=b_tilde, \
                                             A_hat=A_hat, b_hat=b_hat, diagLamb=self.diagLamb)

        delta_vec = delta_vec.A

        # Check linearization and update vector values
        delta = gtsam.VectorValues()
        for key in range(len(self.spls.key_to_col) - 1):
            col = self.spls.key_to_col[key]
            width = self.spls.key_to_col[key + 1] - col
            delta_key = delta_vec[col:col+width]
            delta.insert(key, delta_key)

        self.spls.delta = delta

        self.estimate = self.spls.theta.retract(self.spls.delta)

# class StateEstimation(ABC):
# 
#     def __init__(self):
#         pass
# 
#     @abstractmethod
#     def update(self, factors, new_factors):
#         raise NotImplementedError
# 
#     # Returns the matrix version of the factor
#     def relinearize_factor(self, factor):
#         return None
# 
# class VIO(StateEstimation):
# 
#     """
#     Only update linear system with most recent factors
#     Marginalize old variables into neg factors
#     """
# 
#     def __init__(self):
#         super().__init__()
# 
#     def update(self, theta, factors, new_factors):
#         pass
# 
# 
# class LocalBA(StateEstimation):
#     """
#     Only update linear system with factors close in time or space
#     Does not marginalize old variables
#     Also, delta will only be applied to most recent variables 
#     """
#     def __init__(self):
#         super().__init__()
# 
#     @abstractmethod
#     def update(self, theta, factors, new_factors):
#         pass
# 
# class fullBA(StateEstimation):
#     """
#     Update linear system with all factors
#     Does not marginalize old variables
#     """
#     def __init__(self):
#         super().__init__()
# 
#     @abstractmethod
#     def update(self, theta, factors, new_factors):
#         # 1. Go through all factors, for each factor
#         #   1) If nonlinear factor, relinearize and write to A_tilde
#         # 2. Update diagonal lambda values (if using LM)
#         # 3. Call incremental_opt to get delta
#         # 4. Update theta with delta
# 
#         pass
# 
# class PGO(StateEstimation):
#     """
#     Use pose graph dataset and update all the factors related to poses
#     Does not marginalize old variables
#     """
#     def __init__(self):
#         super().__init__()
# 
#     @abstractmethod
#     def update(self, theta, factors, new_factors):
#         pass
