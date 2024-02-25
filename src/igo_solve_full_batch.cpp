#include <igo.h>

static int igo_reset_A_staged_diff (
    /* --- in/out --- */
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    /* ------------- */
    igo_common* igo_cm
) {
    *num_staged_cols = 0;
    memset(A_staged_diff->data, 0, A_staged_diff->len * sizeof(double));
    return 1;
}

int igo_solve_full_batch (
    /* --- input --- */
    igo_sparse* A,
    igo_dense* b,
    /* --- in/out --- */
    igo_factor** L_handle,
    igo_dense** y_handle,
    igo_dense** x_handle,
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    /* ------------- */
    igo_solve_context* cxt,
    igo_common* igo_cm
) {
    // Convenience variables
    double alpha[2] = {1, 1};
    double beta[2] = {0, 0};
    igo_factor* L = NULL;
    igo_dense* y = NULL;
    igo_dense* x = NULL;

    // All allocated memory
    igo_dense* PAb = NULL;

    igo_free_factor(L_handle, igo_cm);
    igo_free_dense(y_handle, igo_cm);
    igo_free_dense(x_handle, igo_cm);

    // FB0. Analyze and factorize to get LL^T = (P_L * A) (P_L * A)^T
    printf("Before FB0\n");
    *L_handle = igo_analyze_and_factorize(A, igo_cm);
    L = *L_handle;

    // FB1. Compute PAb = PL * A * b
    // printf("Before FB0\n");
    PAb = igo_zeros(cxt->h_hat, 1, CHOLMOD_REAL, igo_cm);
    igo_sdmult(A, 0, alpha, beta, b, PAb, igo_cm);
    igo_permute_rows_dense(PAb, (int*) L->L->Perm, igo_cm);

    // FB2. Solve Ly = PAb. Need to solve for y for future incremental updates
    // printf("Before FB2\n");
    *y_handle = igo_solve(CHOLMOD_L, L, PAb, igo_cm);
    y = *y_handle;

    // FB3. Solve DLt Px = y and unpermute x = P^-1 Px
    // printf("Before FB3\n");
    *x_handle = igo_solve(CHOLMOD_DLt, L, y, igo_cm);
    x = *x_handle;
    igo_unpermute_rows_dense(x, (int*) L->L->Perm, igo_cm);

    // FB4. Reset A_staged_diff
    // printf("Before FB4\n");
    igo_reset_A_staged_diff(num_staged_cols, A_staged_diff, igo_cm);
    
    // Clean up allocated memory
    igo_free_dense(&PAb, igo_cm);

    return 1;
}
