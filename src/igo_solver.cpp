#include <igo.h>
#include <assert.h>

/* Computes Y = alpha (A * A^T - A_neg * A_neg^T) * X + beta Z. 
 * AA^T - A_neg A_neg^T must be positive definite
 * Assume all sizes are correctly allocated. Overwrites Y but does not overwrite X or Z
 * 1. Allocate W 
 * 2. Compute W = A^T * X + 0 * W
 * 3. Compute Y = alpha A * W + 0 * Y
 * 4. Compute W = A_neg^T * X + 0 * W
 * 5. Compute Y = -alpha A_neg * W + 1 * Y
 * 6. Compute Y = Y + beta Z
 * 7. Clean up W1
 * */
static int igo_apply_operator (
    /* --- input --- */
    igo_sparse* A, 
    igo_sparse* A_neg, 
    double* alpha,
    double* beta,
    igo_dense* X,
    igo_dense* Z,
    /* --- output --- */
    igo_dense* Y,
    /* ------------- */
    igo_common* igo_cm
) {

    printf("operator -1\n");
    int m = A->A->nrow;
    int n = A->A->ncol;

    assert(X->B->nrow == m);
    assert(X->B->ncol == 1);
    if(Z != NULL) {
        assert(Z->B->nrow == m);
        assert(Z->B->ncol == 1);
    }
    assert(Y->B->nrow == m);
    assert(Y->B->ncol == 1);

    double alpha_neg[2] = {-alpha[0], -alpha[1]};
    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};

    igo_dense* W = igo_allocate_dense(n, 1, n, igo_cm);

    printf("operator 0\n");
    igo_sdmult(A, 1, alpha_one, alpha_zero, X, W, igo_cm);
    printf("operator 1\n");
    igo_sdmult(A, 0, alpha, alpha_zero, W, Y, igo_cm);

    if(A_neg) {
    printf("operator 2\n");
        igo_sdmult(A_neg, 1, alpha_one, alpha_zero, X, W, igo_cm);
    printf("operator 3\n");
        igo_sdmult(A_neg, 0, alpha_neg, alpha_one, W, Y, igo_cm);
    }

    printf("operator 4\n");
    if(Z != NULL && beta[0] != 0) {
        double* Yx = (double*) Y->B->x;
        double* Zx = (double*) Z->B->x;
        for(int i = 0; i < m; i++) {
            *Yx += beta[0] * (*Zx);
            Yx++;
            Zx++;
        }
    }
    printf("operator 5\n");

    igo_free_dense(&W, igo_cm);

    return 1;
}

/* For a given factor LD (D is stored on the diagonal of L)
 * Solve L * sqrt(D) X = B or sqrt(D) L^T X = B depending on transpose
 * */
static igo_dense* igo_apply_preconditioner (
    /* --- input --- */
    igo_factor* L,
    int transpose,
    igo_dense* B,
    /* ------------- */
    igo_common* igo_cm
) {
    int n = L->L->n;
    assert(B->B->nrow == n);
    assert(B->B->ncol == 1);
    assert(L->L->is_ll == false);

    igo_dense* X = NULL;
    if(transpose) {
        igo_dense* Y = igo_allocate_dense(n, 1, n, igo_cm);
        int* Lp = (int*) L->L->p;
        double* Lx = (double*) L->L->x;
        double* Yx = (double*) Y->B->x;
        double* Bx = (double*) B->B->x;
        for(int i = 0; i < n; i++) {
            double Di = Lx[Lp[i]];
            Yx[i] = Bx[i] / sqrt(Di);
        }
        X = igo_solve(CHOLMOD_Lt, L, Y, igo_cm);
        igo_free_dense(&Y, igo_cm);
    }
    else {
        X = igo_solve(CHOLMOD_L, L, B, igo_cm);
        igo_print_factor(3, "L before solve", L, igo_cm);
        igo_print_dense(3, "B before solve", B, igo_cm);
        igo_print_dense(3, "X after solve", X, igo_cm);
        int* Lp = (int*) L->L->p;
        double* Lx = (double*) L->L->x;
        double* Xx = (double*) X->B->x;
        for(int i = 0; i < n; i++) {
            double Di = Lx[Lp[i]];
            Xx[i] /= sqrt(Di);
        }
    }
    
    return X;
}

static double inner_prod(igo_dense* x, igo_dense* y) {
    int n = x->B->nrow;

    assert(x->B->ncol == 1);
    assert(y->B->nrow == n);
    assert(y->B->ncol == 1);

    double* xx = (double*) x->B->x;
    double* yx = (double*) y->B->x;

    double sum = 0;
    for(int i = 0; i < n; i++) {
        sum += (*xx) * (*yx);
        xx++;
        yx++;
    }

    return sum;
}

/* Compute y = ax + y
 * Overwrites y
 * */
static int daxpy(double a, igo_dense* x, igo_dense* y) {
    int n = x->B->nrow;

    assert(x->B->ncol == 1);
    assert(y->B->nrow == n);
    assert(y->B->ncol == 1);

    double* xx = (double*) x->B->x;
    double* yx = (double*) y->B->x;

    for(int i = 0; i < n; i++) {
        (*yx) += a * (*xx);
        xx++;
        yx++;
    }

    return 1;
}

/* Compute y = ax + by
 * Overwrites y
 * */
static int daxpby(double a, igo_dense* x, double b, igo_dense* y) {
    int n = x->B->nrow;

    assert(x->B->ncol == 1);
    assert(y->B->nrow == n);
    assert(y->B->ncol == 1);

    double* xx = (double*) x->B->x;
    double* yx = (double*) y->B->x;

    for(int i = 0; i < n; i++) {
        *yx = a * (*xx) + b * (*yx);
        xx++;
        yx++;
    }

    return 1;
}

/* Solve (AA^T - A_negA_neg^T)x = b
 * H = AA^T - A_negA_neg^T must be SPD
 * M is the preconditioner
 * x is the initial guess and will store the output
 * Returns 1 if successful.
 * */
int igo_solve_pcgne(
    /* --- input --- */
    igo_sparse* A, 
    igo_sparse* A_neg, 
    igo_dense* b, 
    igo_factor* M,
    double rtol,
    double atol,
    int max_iter,
    /* --- output --- */
    igo_dense* x, 
    igo_pcg_context* cxt,
    /* ------------- */
    igo_common* igo_cm
) {
    // Compute r0 = b - Ax0. Overwrites b
    double alpha_one[2] = {1, 1};
    double alpha_negone[2] = {-1, -1};
    double alpha_zero[2] = {-1, -1};

    int m = A->A->nrow;
    int n = A->A->ncol;

    igo_dense* r = NULL;   // r stores \hat{r}_j
    igo_dense* r0 = igo_allocate_dense(m, 1, m, igo_cm);  
    igo_dense* p = NULL;
    igo_dense* Hp = igo_allocate_dense(m, 1, m, igo_cm);
    igo_dense* MinvHp = NULL;
    igo_dense* MinvTr = NULL;

    // r0 = b - H x0
    printf("before 1\n");
    igo_print_dense(3, "b", b, igo_cm);
    igo_print_dense(3, "x0", x, igo_cm);
    igo_apply_operator(A, A_neg, alpha_negone, alpha_one, x, b, r0, igo_cm);

    // r0_hat = M^(-1) r0
    printf("before 2\n");
    igo_print_dense(3, "r0", r0, igo_cm);
    r = igo_apply_preconditioner(M, 0, r0, igo_cm);
    
    // p0 = M^(-T) r0_hat
    printf("before 3\n");
    igo_print_dense(3, "r", r, igo_cm);
    p = igo_apply_preconditioner(M, 1, r, igo_cm);
    
    int num_iter = 0;
    double r_norm2 = 0, x_norm2 = 0;
    while(1) {
        // r_norm2 = <r_j, r_j>
        printf("before 4\n");
        r_norm2 = inner_prod(r, r);
        printf("rtol = %f\n", r_norm2);
        if(r_norm2 < atol) {
            break;
        }
        x_norm2 = inner_prod(x, x);
        printf("xnorm2 = %f\n", x_norm2);

        if(x_norm2 != 0 && r_norm2 / x_norm2 < rtol) {
            break;
        }
        if(num_iter >= max_iter) {
            break;
        }

        // Hp = H * p_j
        printf("before 5\n");
        igo_apply_operator(A, A_neg, alpha_one, alpha_zero, p, NULL, Hp, igo_cm);

        // alpha_j = <\hat{r}_j, \hat{r}_j> / <p_j, Hp>
        printf("before 6\n");
        double p_normH = inner_prod(p, Hp);
        double alpha_j = r_norm2 / p_normH;

        // x_j+1 = x_j + alpha_j p_j
        printf("before 7\n");
        daxpy(alpha_j, p, x);

        // r_j+1 = r_j - alpha_j M^{-1} Hp
        printf("before 8\n");
        igo_free_dense(&MinvHp, igo_cm);
        MinvHp = igo_apply_preconditioner(M, 0, Hp, igo_cm);
        daxpy(-alpha_j, MinvHp, r);

        // beta_j = <r_j+1, r_j+1> / <r_j, r_j>
        printf("before 9\n");
        double beta_j = inner_prod(r, r) / r_norm2;

        // pj+1 = M^{-T}r_j+1 + beta_j p_j
        printf("before 10\n");
        igo_free_dense(&MinvTr, igo_cm);
        MinvTr = igo_apply_preconditioner(M, 1, r, igo_cm);
        daxpby(1, MinvTr, beta_j, p);

        num_iter++;
    }

    cxt->aerr = r_norm2;
    cxt->rerr = x_norm2 != 0? r_norm2 / x_norm2 : 0;
    cxt->num_iter = num_iter;

    igo_free_dense(&r, igo_cm);
    igo_free_dense(&r0, igo_cm);
    igo_free_dense(&p, igo_cm);
    igo_free_dense(&Hp, igo_cm);
    igo_free_dense(&MinvHp, igo_cm);
    igo_free_dense(&MinvTr, igo_cm);

    return 1;
}
