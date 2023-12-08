from igo_base import IgoBase

class IgoSelectiveCholeskyUpdate(IgoBase):
    """
    SelectiveCholeskyUpdate picks the high norm rows to update Cholesky with and use 
    as a preconditioner in PCG
    """
    id_string = "selcholupdate"
    def __init__(self, params):
        super().__init__(params)
        self.H = None
        self.L = None

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

        # 1. Find high norm rows (factors) in A_tilde and replace corresponding rows (factors) of A
        # 2. Add A_hat to A
        # 3. Compute L as cholesky_AAt(A)
        # 4. Now replace the all rows of A with A_tilde 
        # 5. Permute columns of A with P
        # 6. Use conjugate gradient to solve A^TAx = A^Tb

        print("cp1")
        if self.A.shape[0] > 0:
            print(self.A.shape, A_tilde.shape)
            print("A(0, 0) = ", self.A[0, 0])
            print("H(0, 0) = ", self.H[0, 0])

        # 1. Find high norm rows (factors) in A_tilde and replace corresponding rows (factors) of A
        A_tilde_rows = np.array(list(set(A_tilde.nonzero()[0])))
        b_tilde_rows = np.array(list(set(b_tilde.nonzero()[0])))
        if len(A_tilde_rows) > 0:
            print(self.A.shape, A_tilde.shape, A_tilde_rows.shape)
            A_tilde_diff = self.A[A_tilde_rows] - A_tilde[A_tilde_rows]
            row_norms = scipy.sparse.linalg.norm(A_tilde_diff, ord=float('inf'), axis=1)
            A_row_norms = scipy.sparse.linalg.norm(A_tilde[A_tilde_rows], ord=float('inf'), axis=1)

            # threshold = 0.004
            # max_val = A_row_norms.max()

            # high_rows = A_tilde_rows[np.where(row_norms > threshold)]
            # high_rows = A_tilde_rows[np.where(row_norms > threshold * A_row_norms)]
            # high_rows = A_tilde_rows[np.where(row_norms > threshold * max_val)]

            percent_rows = 0.05
            num_rows = int(0.05 * len(A_tilde_rows))
            high_rows = A_tilde_rows[np.argsort(row_norms)[-num_rows:]]

            print("num relin rows = ", len(high_rows))
            # print(row_norms[high_rows])
            # print(A_row_norms[high_rows])

            # DEBUG
            A_correct = deepcopy(self.A)
            A_correct[A_tilde_rows] = A_tilde[A_tilde_rows]
            # DEBUG END

            # print(high_rows, self.A.shape, A_tilde.shape)
            print(A_tilde[0, :], row_norms[0])
            self.A[high_rows] = A_tilde[high_rows]
            print(self.A.shape, A_tilde.shape)

            # DEBUG
            print("max_diff = ", np.max((A_correct - self.A).A))

        print("cp2")
        if self.A.shape[0] > 0:
            print("A(0, 0) = ", self.A[0, 0])
            print("H(0, 0) = ", self.H[0, 0])

        # 2. Add A_hat to A
        self.A.resize((resize_rows, resize_cols))
        self.negA.resize((resize_rows, resize_cols))
        self.b.resize((resize_rows, 1))
        self.negb.resize((resize_rows, 1))

        self.diagLamb.resize((resize_cols, resize_cols))
        self.diagLamb[diagLamb.nonzero()] = diagLamb[diagLamb.nonzero()]

        # Assign rows of A_hat to A. Need this to keep 0 entries
        A_hat_rows = np.array(list(set(A_hat.nonzero()[0])))
        b_hat_rows = np.array(list(set(b_hat.nonzero()[0])))
        if len(A_hat.nonzero()[0]) > 0:
            self.A[A_hat_rows] = A_hat[A_hat_rows]
            self.b[b_hat_rows] = b_hat[b_hat_rows]
        
        # 3. Compute L as cholesky(H). Technically H does not need to be formed 
        self.H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb.T @ self.diagLamb
        factor = cholesky(self.H)
        self.L = factor.L()
        self.P = factor.P()

        print("cp3")
        if self.A.shape[0] > 0:
            print("A(0, 0) = ", self.A[0, 0])
            print("H(0, 0) = ", self.H[0, 0])

        # 4. Now replace the all rows of A with A_tilde 
        if A_tilde.shape[0] != 0 and A_tilde.shape[1] != 0:
            print(self.A.shape, A_tilde.shape)
            self.A[A_tilde_rows, 0:A_tilde.shape[1]] = A_tilde[A_tilde_rows]
            self.b[b_tilde_rows] = b_tilde[b_tilde_rows]

        print("cp4")
        if self.A.shape[0] > 0:
            print("A(0, 0) = ", self.A[0, 0])
            print("H(0, 0) = ", self.H[0, 0])

        # 5. Permute columns of A with P
        AP = self.A[:, self.P]
        negAP = self.negA[:, self.P]
        sqrtLambP = self.diagLamb[self.P][:, self.P]

        print("cp5")
        print("A(0, 0) = ", self.A[0, 0])
        print("H(0, 0) = ", self.H[0, 0])

        print(AP.shape, self.b.shape, negAP.shape, self.negb.shape, sqrtLambP.shape)
        rhs = AP.T @ self.b - negAP.T @ self.negb

        print(rhs.shape)

        linOps = applyATA(AP, negAP, sqrtLambP, self.L)

        CGcallback.reset()
        x, info = cg(linOps, rhs.A, callback=CGcallback.callback, tol=1e-5, maxiter=100)

        # print(x, info)
        print(f"cg iter = {CGcallback.count}")

        self.H = self.A.T @ self.A - self.negA.T @ self.negA + self.diagLamb
        self.Atb = self.A.T @ self.b

        self.factor = cholesky(self.H)
        self.x = self.factor(self.Atb)

        return self.x

    def marginalize(self, cols):
        pass
