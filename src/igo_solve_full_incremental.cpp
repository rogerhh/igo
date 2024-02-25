#include <igo.h>
#include <assert.h>

#define MIN(x, y) (x < y? x : y)
#define MAX(x, y) (x > y? x : y)

int igo_solve_full_incremental (
    /* --- input --- */
    igo_sparse* A,
    igo_dense* b,
    igo_sparse* A_tilde,
    igo_sparse* b_tilde,
    igo_sparse* A_tilde_neg,
    igo_sparse* b_tilde_neg,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- in/out --- */
    igo_sparse* A_staged_neg,
    igo_dense* b_staged_neg,
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
    igo_factor* L = *L_handle;
    igo_dense* y = *y_handle;
    igo_dense* x = NULL;

    igo_free_dense(x_handle, igo_cm);

    // FI0. Resize L and y if needed
    // printf("Before FI0\n");
    if(cxt->h_hat > cxt->h_orig) {
        igo_resize_factor(cxt->h_hat, L->L->nzmax, L, igo_cm);
        igo_resize_dense(cxt->h_hat, 1, cxt->h_hat, y, igo_cm);
    }

    // FI1. Go through columns of A_tilde_neg, if corresponding column in A_staged_neg is 0, replace with column in A_tilde_neg
    // printf("Before FI1\n");
    igo_replace_staged(
            A_tilde, A_tilde_neg, A_staged_neg, 
            b_tilde, b_tilde_neg, b_staged_neg,
            A_staged_diff, num_staged_cols,
            igo_cm);
    cxt->num_relin_staged_cols = *num_staged_cols - cxt->A_hat_nz_cols;

    // FI2. Go through columns of A_staged_neg and compare with corresponding columns in A. Pick the k highest columns of the largest difference. The column indices are in Ck
    // printf("Before FI2\n");
    int percent_sel_cols = ((double) cxt->h_hat) * igo_cm->SEL_COLS_RATE;
    int max_num_sel_cols = MAX(percent_sel_cols, igo_cm->MIN_SEL_COLS);
    cxt->num_sel_cols = 0;
    cxt->sel_cols = (int*) malloc((max_num_sel_cols + cxt->A_hat_nz_cols) * sizeof(int));
    igo_pick_k_highest_diff(
            max_num_sel_cols, cxt->orig_cols, 
            A_staged_diff,
            cxt->num_relin_staged_cols,
            &cxt->num_sel_cols, cxt->sel_cols, 
            igo_cm);

    int num_sel_relin_cols = cxt->num_sel_cols;
    
    // FI3. Get the submatrices A_sel = A[:,Ck], A_sel_neg = A_staged_neg[:,Ck]
    // FI4. Permute PA_sel = P_L * A_sel, PA_sel_neg = P_L * A_sel_neg
    // printf("Before FI3,4\n");

    int* P = (int*) L->L->Perm;
    cxt->PA_sel = igo_submatrix(
                    A, 
                    P, cxt->h_hat,
                    cxt->sel_cols, cxt->num_sel_cols, 
                    true, true, igo_cm);

    cxt->PA_sel_neg = igo_submatrix(
                        A_staged_neg, 
                        P, cxt->h_hat,
                        cxt->sel_cols, cxt->num_sel_cols, 
                        true, true, igo_cm);

    // FI5. Compute PAb = PA_sel * b_sel, PAb_sel_neg = PA_sel_neg * b_sel_neg
    // FI6. Compute PAb_delta = PAb_sel - PAb_sel as a dense vector
    // printf("Before FI5, 6\n");
    cxt->PAb_delta = igo_compute_PAb_delta_sel(
                        cxt->PA_sel, cxt->PA_sel_neg, 
                        b, b_staged_neg,
                        cxt->sel_cols, cxt->num_sel_cols,
                        igo_cm);

    // FI7. Call igo_updown2_solve(PA_sel, PA_sel_neg, L, y, PAb_delta)
    // printf("Before FI6\n");
    igo_updown2_solve(cxt->PA_sel, cxt->PA_sel_neg, L, y, cxt->PAb_delta, igo_cm);

    // FI11. Set A_staged_diff[Ck] = 0 
    // printf("Before FI7\n");
    igo_set_col_zero(
            cxt->sel_cols, cxt->num_sel_cols, 
            A_staged_diff, 
            num_staged_cols,
            igo_cm);

    // igo_check_state1(igo_cm);

    // FI8. Permute PA_hat = P_L * A_hat 
    // printf("Before FI8\n");
    cxt->PA_hat = igo_submatrix(A_hat, P, cxt->h_hat, NULL, -1, true, true, igo_cm);

    // FI9. Compute PAb_hat = A_hat * b_hat
    // printf("Before FI9\n");
    cxt->PAb_hat = igo_ssmult(cxt->PA_hat, b_hat, 0, true, true, igo_cm);
    igo_add_sparse_to_dense(cxt->PAb_hat, 1, cxt->PAb_delta, igo_cm);

    // FI10. Call igo_updown_solve(1, PA_hat, L, y, PAb_hat)
    // printf("Before FI10\n");
    igo_updown_solve(1, cxt->PA_hat, L, y, cxt->PAb_delta, igo_cm);

    igo_set_A_hat_col_zero(
            cxt->orig_cols,
            A_staged_diff, 
            num_staged_cols,
            igo_cm);

    if(*num_staged_cols == 0) {
        // FI12. If A_staged_neg == 0, solve DLtx = y and unpermute x
        // printf("Before FI12\n");
        *x_handle = igo_solve(CHOLMOD_DLt, L, y, igo_cm);
        x = *x_handle;
        igo_unpermute_rows_dense(x, (int*) L->L->Perm, igo_cm);
    }
    else {
        // FI13. Else, first get initial guess. DLt Px = y
        // printf("Before FI13\n");
        *x_handle = igo_solve(CHOLMOD_DLt, L, y, igo_cm);
        x = *x_handle;

        // FI14. solve PCGNE PAA^TP^TPx = PAb
        // printf("Before FI14\n");

        double alpha_one[2] = {1, 1};
        double alpha_zero[2] = {0, 0};
        cxt->PA = igo_submatrix(A, P, cxt->h_hat, NULL, -1, true, true, igo_cm);
        cxt->PAb = igo_zeros(cxt->h_hat, 1, CHOLMOD_REAL, igo_cm);
        igo_sdmult(cxt->PA, 0, alpha_one, alpha_zero, b, cxt->PAb, igo_cm);

        cxt->pcg_cxt = (igo_pcg_context*) malloc(sizeof(igo_pcg_context));
        igo_solve_pcgne(
                cxt->PA, NULL, cxt->PAb, L, 
                igo_cm->pcg_rtol, igo_cm->pcg_atol, cxt->h_hat, 
                x, cxt->pcg_cxt, 
                igo_cm);

        printf("A_tilde_nzcol: %d\n", cxt->A_tilde_nz_cols);
        printf("A_staged_neg nzcol: %d\n", cxt->num_relin_staged_cols);
        printf("Selected relin cols count: %d\n", num_sel_relin_cols);
        printf("num iter: %d\n", cxt->pcg_cxt->num_iter);

        igo_unpermute_rows_dense(x, (int*) L->L->Perm, igo_cm);

    }
    return 1;
}

