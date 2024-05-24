from common_packages import *

from igo_preconditioner_generator_base import IgoPreconditionerGeneratorBase
from igo_row_selector import igo_row_selector_types

class IgoSelChol(IgoPreconditionerGeneratorBase):
    """
    Base class for all selective cholesky update preconditioning.
    Pick a set of rows in A to replace in chol_A
    Has a IgoRowSelectorBase object to select rows
    """

    id_string = "selchol"

    def __init__(self, params):
        super().__init__(params)

        self.row_selector = None
        self.chol_A = csr_matrix(([], ([], [])), shape=(0, 0))
        self.chol_C = csr_matrix(([], ([], [])), shape=(0, 0))
        self.chol_sqrtLamb = csr_matrix(([], ([], [])), shape=(0, 0))

        for row_selector in igo_row_selector_types:
            if row_selector.id_string == params["iterative"]["selchol"]["row_selector"]:
                self.row_selector = row_selector(params)
                break

    # if setting up LC step, pick all rows
    def generate_preconditioner(self, params, **kwargs):
        A = kwargs["A"]
        A_tilde = kwargs["A_tilde"]
        A_hat = kwargs["A_hat"]
        C = kwargs["C"]
        C_tilde = kwargs["C_tilde"]
        C_hat = kwargs["C_hat"]
        sqrtLamb = kwargs["sqrtLamb"]
        sqrtLamb_tilde = kwargs["sqrtLamb_tilde"]
        sqrtLamb_hat = kwargs["sqrtLamb_hat"]

        # First resize chol_A, chol_C, chol_sqrtLamb to accomodate the new values
        new_A_rows, new_A_cols = A.shape
        new_C_rows, new_C_cols = C.shape
        new_sqrtLamb_rows, new_sqrtLamb_cols = sqrtLamb.shape

        self.chol_A.resize((new_A_rows, new_A_cols))
        self.chol_C.resize((new_C_rows, new_C_cols))
        self.chol_sqrtLamb.resize((new_sqrtLamb_rows, new_sqrtLamb_cols))

        if "setup_lc_step" in params.keys() and params["setup_lc_step"]:
            A_sel_rows = np.arange(A.shape[0])
            C_sel_rows = np.arange(C.shape[0])
            sqrtLamb_sel_rows = np.arange(sqrtLamb.shape[0])

        else:
            A_sel_rows, C_sel_rows, sqrtLamb_sel_rows = \
                        self.row_selector.select_rows(
                            params=params,
                            A=A, A_tilde=A_tilde, A_hat=A_hat, 
                            C=C, C_tilde=C_tilde, C_hat=C_hat,
                            sqrtLamb=sqrtLamb, sqrtLamb_tilde=sqrtLamb_tilde, sqrtLamb_hat=sqrtLamb_hat,
                            chol_A=self.chol_A, 
                            chol_C=self.chol_C, 
                            chol_sqrtLamb=self.chol_sqrtLamb)

        # Replace rows in chol_A, chol_C, chol_sqrtLamb to rows in A_hat, C_hat, sqrtLamb_hat
        self.chol_A[A_hat.nonzero()] = A_hat[A_hat.nonzero()]
        self.chol_C[C_hat.nonzero()] = C_hat[C_hat.nonzero()]
        self.chol_sqrtLamb[sqrtLamb_hat.nonzero()] = sqrtLamb_hat[sqrtLamb_hat.nonzero()]

        if len(A_sel_rows) > 0:
            self.chol_A[A_sel_rows] = A[A_sel_rows]

        if len(C_sel_rows) > 0:
            self.chol_C[C_sel_rows] = C[C_sel_rows]

        if len(sqrtLamb_sel_rows) > 0:
            self.chol_sqrtLamb[sqrtLamb_sel_rows] = sqrtLamb[sqrtLamb_sel_rows]

        H = self.chol_A.T @ self.chol_A - self.chol_C.T @ self.chol_C + self.chol_sqrtLamb.T @ self.chol_sqrtLamb

        factor = cholesky(H)
        L = factor.L()
        P = factor.P()

        return L, P

