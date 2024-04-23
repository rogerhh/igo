from common_packages import *

from igo_base import IgoBase

class IgoIterativeSolver(IgoBase):

    """
    Driver class of all iterative methods 
    Each class should have a preconditioner generator and a solver
    The generator and the solver can be further derived from
    """

    id_string = "iterative"

    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.M = None
        self.P = None

        # preconditioner_generator generates a preconditioner and updates H, L
        # solver uses the preconditioner and solves the problem
        self.precond_gen = None
        self.solver = None

        for precond_type in igo_precond_types:
            if params["igo_precond_id"] == precond_type.id_string:
                self.precond_gen = precond_type(params)

        for iter_type in igo_iter_solver_types:
            if params["igo_iter_solver_id"] == iter_type.id_string:
                self.solver = iter_type(params)

        assert(self.precond_gen is not None)
        assert(self.solver is not None)

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
        old_rows, old_cols = self.A.shape
        new_rows, new_cols = A_hat.shape
        resize_cols = new_cols if new_cols > old_cols else old_cols
        resize_rows = new_rows if new_rows > old_rows else old_rows

        # Do all the necessary resizing here
        self.A.resize((resize_rows, resize_cols))
        self.negA.resize((resize_rows, resize_cols))
        self.b.resize((resize_rows, 1))
        self.negb.resize((resize_rows, 1))
        self.diagLamb.resize((resize_cols, resize_cols))
        self.diagLamb[diagLamb.nonzero()] = diagLamb[diagLamb.nonzero()]
        A_tilde.resize((resize_rows, resize_cols))
        b_tilde.resize((resize_rows, 1))

        # 1. Replace all rows in A with corresponding rows of A_tilde
        # 2. Compare A to chol_A and pick high norm rows
        # 3. Add A_hat to A and chol_A
        # 4. Compute L as cholesky_AAt(chol_A.T)
        # 5. Permute columns of A with P
        # 6. Use conjugate gradient to solve A^TAx = A^Tb

        A_tilde_rows = np.array(list(set(A_tilde.nonzero()[0])))
        b_tilde_rows = np.array(list(set(b_tilde.nonzero()[0])))
        A_hat_rows = np.array(list(set(A_hat.nonzero()[0])))
        b_hat_rows = np.array(list(set(b_hat.nonzero()[0])))

        # 1. Replace all rows in A with corresponding rows of A_tilde
        if A_tilde.shape[0] != 0 and A_tilde.shape[1] != 0:
            log("A shapes = ", self.A.shape, len(A_tilde_rows))
            self.A[A_tilde_rows, 0:A_tilde.shape[1]] = A_tilde[A_tilde_rows]
            self.b[b_tilde_rows] = b_tilde[b_tilde_rows]

        self.L, self.P = self.precond_gen.generate(self, old_cols, new_cols, A_tilde_rows, b_tilde_rows, A_hat_rows, b_hat_rows, params)


        exit(1)

        if len(A_hat_rows) > 0:
            self.A[A_hat_rows] = A_hat[A_hat_rows]
            self.b[b_hat_rows] = b_hat[b_hat_rows]

        # 5. Permute columns of A with P
        AP = self.A[:, self.P]
        negAP = self.negA[:, self.P]
        diagLambP = self.diagLamb[self.P][:, self.P]

        log(AP.shape, self.b.shape, negAP.shape, self.negb.shape, diagLambP.shape)
        rhs = AP.T @ self.b - negAP.T @ self.negb
        rhs = spsolve_triangular(self.L, rhs.A, lower=True)
        print(rhs.shape)

        # 6. Use conjugate gradient to solve A^TAx = A^Tb
        linOps = applyPreconditionedATA(AP, negAP, diagLambP, self.L)
        CGcallback.reset()
        LTxP, info = cg(linOps, rhs, callback=CGcallback.callback, tol=params["cg_tolerance"], maxiter=params["max_cg_iter"])
        LTxP = LTxP.reshape((-1, 1))

        xP = spsolve_triangular(self.L.T, LTxP, lower=False)

        self.x = np.zeros_like(xP)
        self.x[self.P] = xP
        
        print(f"cg iter = {CGcallback.count}, info = {info}")
        if info != 0:
            print("CG failed to converge!")
            assert("setup_lc_step" not in params.keys() or not params["setup_lc_step"])
            H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb
            Atb = self.A.T @ self.b

            factor = cholesky(H)
            self.x = factor(Atb)

        print(f"step = ", params["step"], params["outer_iter"], params["profile_outer_iter"], params["lc_steps"])
        if params["step"] in params["lc_steps"] and params["outer_iter"] == params["profile_outer_iter"]:
            print("adding to yaml")
            step = params["step"]
            params["output_yaml_obj"][step]["A_height"] = self.A.shape[0]
            params["output_yaml_obj"][step]["A_width"] = self.A.shape[1]
            # w = np.real(scipy.sparse.linalg.eigs(applyPreconditionedATA(AP, negAP, diagLambP, self.L), k=self.A.shape[1] - 2, return_eigenvectors=False))
            # params["output_yaml_obj"][step]["cond"] = (w[-1] / w[0]).item()
            params["output_yaml_obj"][step]["cg_iter"] = CGcallback.count
            params["output_yaml_obj"][step]["100th"] = np.percentile(np.abs(self.x), 100).item()
            params["output_yaml_obj"][step]["75th"] = np.percentile(np.abs(self.x), 75).item()
            params["output_yaml_obj"][step]["50th"] = np.percentile(np.abs(self.x), 50).item()
            params["output_yaml_obj"][step]["25th"] = np.percentile(np.abs(self.x), 20).item()
            print(np.max(np.abs(self.x)))
            # params["output_yaml_obj"][step]["evals"] = w.tolist()



        # if self.H.shape[0] == 1224:
        #     w_unconditioned = scipy.sparse.linalg.eigs(applyATA(AP, negAP, diagLambP), k=1222, return_eigenvectors=False)
        #     w_preconditioned = scipy.sparse.linalg.eigs(applyPreconditionedATA(AP, negAP, diagLambP, self.L), k=1222, return_eigenvectors=False)
        #     print(w_unconditioned)
        #     plt.plot(range(len(w_unconditioned)), w_unconditioned)
        #     plt.plot(range(len(w_preconditioned)), w_preconditioned)
        #     plt.yscale("log")
        #     plt.show()
        #     exit(0)
        # self.H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb
        # self.Atb = self.A.T @ self.b

        # self.factor = cholesky(self.H)
        # self.x = self.factor(self.Atb)

        return self.x

    def marginalize(self, cols):
        pass



