#include <igo.h>
#include <ccolamd.h>
#include <assert.h>

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

    int nrow = A->A->nrow;
    int ncol = A->A->ncol;
    int* Ap = (int*) A->A->p;
    int* Ai = (int*) A->A->i;
    igo_AT_pattern* AT = igo_cm->AT;

    const int Alen = ccolamd_recommended(Ap[ncol], ncol, nrow);

    // All allocated memory
    igo_sparse* H = NULL;
    igo_dense* PAb = NULL;
    int* new_p = (int*) malloc((cxt->h_hat + 1) * sizeof(int));
    int* new_i = (int*) malloc(Alen * sizeof(int));
    int* cmember = (int*) malloc(cxt->h_hat * sizeof(int));

    memset(new_i, 0, Alen * sizeof(int));

    new_p[0] = 0;
    for(int j = 0; j < AT->ncol; j++) {
        new_p[j + 1] = new_p[j] + AT->len[j];
        memcpy(new_i + new_p[j], AT->i[j], AT->len[j] * sizeof(int));
    }

    memset(cmember, 0,  cxt->h_hat * sizeof(int));

    if(cxt->h_orig > 0) {
        for(int i = cxt->h_orig; i < cxt->h_hat; i++) {
            cmember[i] = 1;
        }
    }

    igo_free_factor(L_handle, igo_cm);
    igo_free_dense(y_handle, igo_cm);
    igo_free_dense(x_handle, igo_cm);

    // FB0. Analyze and factorize to get LL^T = (P_L * A) (P_L * A)^T
    // printf("Before FB0\n");

    double knobs[CCOLAMD_KNOBS];
    ccolamd_set_defaults(knobs);
    knobs[CCOLAMD_DENSE_ROW] = -1;
    knobs[CCOLAMD_DENSE_COL] = -1;

    int stats[CCOLAMD_STATS]; /* colamd arg 7: colamd output statistics and error codes */

    // printf("Ap: ");
    // for(int i = 0; i <= cxt->h_hat; i++) {
    //     printf("%d ", new_p[i]);
    // }
    // printf("\n");
    // printf("Ai: ");
    // for(int i = 0; i < Alen; i++) {
    //     printf("%d ", new_i[i]);
    // }
    // printf("\n");
    // printf("cmember: ");
    // for(int i = 0; i < cxt->h_hat; i++) {
    //     printf("%d ", cmember[i]);
    // }
    // printf("\n");
    // printf("%d %d %d\n", ncol, nrow, Ap[ncol]);
    // fflush(stdout);

    int rv = ccolamd(ncol, nrow, Alen, new_i, new_p, knobs, stats, cmember);

    assert(rv == 1);

    // H = igo_aat(A, NULL, -1, CHOLMOD_REAL, igo_cm);
    // H->A->stype = 1;

    // *L_handle = igo_analyze_p_and_factorize(H, new_p, NULL, -1, igo_cm);
    *L_handle = igo_analyze_p_and_factorize(A, new_p, NULL, -1, igo_cm);
    // *L_handle = igo_analyze_and_factorize(A, igo_cm);
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
    igo_free_sparse(&H, igo_cm);
    igo_free_dense(&PAb, igo_cm);
    free(new_p);
    free(new_i);
    free(cmember);

    return 1;
}
