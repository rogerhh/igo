from common_packages import *

from igo_base import IgoBase
from igo_preconditioner_generator import igo_preconditioner_types
from igo_iterative_solver import igo_iterative_solver_types

class IgoIterative(IgoBase):
    """
    Compute the optimal solution for the incremental optimization problem using iterative methods
    Iterative solvers should have a precond_gen and a solver
    precond_gen generates the preconditioner 
    solver solves the linear system using an iterative methods with the preconditioner
    """

    id_string = "iterative"

    def __init__(self, params):
        super().__init__(params)

        self.precond_gen = None
        self.solver = None

        for precond_gen in igo_preconditioner_types:
            if precond_gen.id_string == self.params["iterative"]["preconditioner"]:
                self.precond_gen = precond_gen(self.params)
                break

        for solver in igo_iterative_solver_types:
            if solver.id_string == self.params["iterative"]["iterative_solver"]:
                self.solver = solver(self.params)
                break

        print(self.precond_gen)
        assert(self.precond_gen is not None)
        assert(self.solver is not None)
    
    def incremental_opt(self, 
                        A_tilde, b_tilde, 
                        A_hat, b_hat, 
                        C_tilde, d_tilde,
                        C_hat, d_hat,
                        sqrtLamb_tilde,
                        sqrtLamb_hat, 
                        params):

        old_A_rows, old_A_cols = self.A.shape
        new_A_rows, new_A_cols = A_hat.shape
        old_C_rows, old_C_cols = self.C.shape
        new_C_rows, new_C_cols = C_hat.shape
        old_sqrtLamb_rows, old_sqrtLamb_cols = self.sqrtLamb.shape
        new_sqrtLamb_rows, new_sqrtLamb_cols = sqrtLamb_hat.shape

        assert old_A_cols == old_C_cols == old_sqrtLamb_cols
        assert new_A_cols == new_C_cols == new_sqrtLamb_cols

        # Resize the matrices to accomodate the new values
        resize_A_cols = new_A_cols if new_A_cols > old_A_cols else old_A_cols
        resize_A_rows = new_A_rows if new_A_rows > old_A_rows else old_A_rows
        resize_C_cols = new_C_cols if new_C_cols > old_C_cols else old_C_cols
        resize_C_rows = new_C_rows if new_C_rows > old_C_rows else old_C_rows
        resize_sqrtLamb_cols = new_sqrtLamb_cols if new_sqrtLamb_cols > old_sqrtLamb_cols else old_sqrtLamb_cols
        resize_sqrtLamb_rows = new_sqrtLamb_rows if new_sqrtLamb_rows > old_sqrtLamb_rows else old_sqrtLamb_rows

        self.A.resize((resize_A_rows, resize_A_cols))
        self.b.resize((resize_A_rows, 1))
        self.C.resize((resize_C_rows, resize_C_cols))
        self.d.resize((resize_C_rows, 1))
        self.sqrtLamb.resize((resize_sqrtLamb_rows, resize_sqrtLamb_cols))

        # Replace entries in A, b, C, d, sqrtLamb with the new values
        self.A[A_tilde.nonzero()] = A_tilde[A_tilde.nonzero()]
        self.b[b_tilde.nonzero()] = b_tilde[b_tilde.nonzero()]
        self.C[C_tilde.nonzero()] = C_tilde[C_tilde.nonzero()]
        self.d[d_tilde.nonzero()] = d_tilde[d_tilde.nonzero()]
        self.sqrtLamb[sqrtLamb_tilde.nonzero()] = sqrtLamb_tilde[sqrtLamb_tilde.nonzero()]

        # Insert the new values
        self.A[A_hat.nonzero()] = A_hat[A_hat.nonzero()]
        self.b[b_hat.nonzero()] = b_hat[b_hat.nonzero()]
        self.C[C_hat.nonzero()] = C_hat[C_hat.nonzero()]
        self.d[d_hat.nonzero()] = d_hat[d_hat.nonzero()]
        self.sqrtLamb[sqrtLamb_hat.nonzero()] = sqrtLamb_hat[sqrtLamb_hat.nonzero()]

        # Generate the preconditioner
        M, P = self.precond_gen.generate_preconditioner(
                params=params,
                A=self.A, A_tilde=A_tilde, A_hat=A_hat,
                C=self.C, C_tilde=C_tilde, C_hat=C_hat,
                sqrtLamb=self.sqrtLamb, sqrtLamb_tilde=sqrtLamb_tilde, sqrtLamb_hat=sqrtLamb_hat)

        # Solve the linear system
        x = self.solver.solve(self.A, self.b, self.C, self.d, self.sqrtLamb, M, P, params)

        return x

    def marginalize(self, keys):
        raise NotImplementedError("marginalize not implemented")




