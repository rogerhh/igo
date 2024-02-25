#include <igo.h>
#include <assert.h>

// One way of comparing the vector a and b is to give the maximum absolute difference
static double igo_column_diff1(double* a, double* b, int len) {
    double max = 0;
    for(int i = 0; i < len; i++) {
        double diff = fabs(a[i] - b[i]);
        if(max < diff) {
            max = diff;
        }
    }
    return max;
}


int igo_replace_staged(
    /* --- input --- */
    igo_sparse* A_tilde, 
    igo_sparse* A_tilde_neg, 
    igo_sparse* A_staged_neg, 
    igo_sparse* b_tilde,
    igo_sparse* b_tilde_neg,
    igo_dense* b_staged_neg,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* ------------- */
    igo_common* igo_cm
) {
    if(A_tilde_neg == NULL) { return 1; }

    igo_check_invariant_sparse(A_staged_neg, igo_cm);
    igo_check_invariant_sparse(A_tilde_neg, igo_cm);

    int ncol = A_tilde_neg->A->ncol;

    assert(b_tilde->A->nrow == ncol);
    assert(b_tilde->A->ncol == 1);
    assert(b_staged_neg->B->nrow >= ncol);
    assert(b_staged_neg->B->ncol == 1);

    int* A_tilde_p = (int*) A_tilde->A->p;
    double* A_tilde_x = (double*) A_tilde->A->x;
    int* A_tilde_neg_p = (int*) A_tilde_neg->A->p;
    double* A_tilde_neg_x = (double*) A_tilde_neg->A->x;
    int* A_staged_neg_p = (int*) A_staged_neg->A->p;
    double* A_staged_neg_x = (double*) A_staged_neg->A->x;
    int* b_tilde_i = (int*) b_tilde->A->i;
    double* b_tilde_x = (double*) b_tilde->A->x;
    double* b_tilde_neg_x = (double*) b_tilde_neg->A->x;
    double* b_staged_neg_x = (double*) b_staged_neg->B->x;
    
    int count = 0;
    for(int j = 0; j < ncol; j++) {
        int nz1 = A_tilde_neg_p[j + 1] - A_tilde_neg_p[j];
        double staged = A_staged_diff->data[j];

        if(nz1 == 0) { continue; }

        int p0 = A_tilde_p[j];
        int p1 = A_tilde_neg_p[j];
        int p2 = A_staged_neg_p[j];
        int bi = b_tilde_i[count];

        assert(bi == j);

        // Only copy old values in if it had not been updated in L
        if(staged == 0) {
            memcpy(A_staged_neg_x + p2, A_tilde_neg_x + p1, nz1 * sizeof(double));
            b_staged_neg_x[j] = b_tilde_neg_x[count];

            (*num_staged_cols)++;
        }

        double A_diff = igo_column_diff1(A_staged_neg_x + p1, A_tilde_x + p0, nz1);
        double b_diff = fabs(b_staged_neg_x[j] - b_tilde_neg_x[count]);

        A_staged_diff->data[j] = A_diff > b_diff? A_diff : b_diff;

        count++;
    }

    return 1;
}

// Return a list of indices to at most max_k columns with the highest 
// difference in A and A_staged_neg
// Assume indices is already allocated
// Only consider the first ncol columns. The columns after the first ncol are guaranteed to be picked
int igo_pick_k_highest_diff(
    /* --- inputs --- */
    int max_k,
    int ncol,
    igo_vector_double* A_staged_diff,
    int num_relin_staged_cols,
    /* --- outputs --- */
    int* k,
    int* indices,
    /* --- common --- */
    igo_common* igo_cm
) {

    *k = 0;

    if(max_k == 0) { return 1; }

    if(max_k >= num_relin_staged_cols) {
        for(int i = 0; i < ncol; i++) {
            double diff = A_staged_diff->data[i];
            if(diff > 0) {
                indices[(*k)++] = i;
            }
        }
    }
    else {

        double* sel_diff = (double*) malloc(max_k * sizeof(double));

        // TODO: Make this algorithm more efficient
        for(int i = 0; i < ncol; i++) {
            double diff = A_staged_diff->data[i];
            if(diff == 0) { continue; }

            if(*k >= max_k && diff <= sel_diff[0]) { continue; }

            int idx = 0;

            if(*k < max_k) {

                for(idx = 0; idx < *k; idx++) {
                    if(diff < sel_diff[idx]) { 
                        break; 
                    }
                }

                // shift entries to the right to make space
                memmove(sel_diff + idx + 1, sel_diff + idx, (*k - idx) * sizeof(double));
                memmove(indices + idx + 1, indices + idx, (*k - idx) * sizeof(int));
                (*k)++;
            }
            else {

                for(idx = 1; idx < *k; idx++) {
                    if(diff < sel_diff[idx]) { 
                        break; 
                    }
                }
                idx--;

                // shift entries to the left to make space
                memmove(sel_diff, sel_diff + 1, idx * sizeof(double));
                memmove(indices, indices + 1, idx * sizeof(int));
            }

            // Shift all memory up to this point over by 1
            sel_diff[idx] = diff;
            indices[idx] = i;
        }
        free(sel_diff);
        sel_diff = NULL;
    }

    // Sort the indices. FIXME: Make this faster
    for(int i = 0; i < *k - 1; i++) {
        for(int j = 0; j < *k - i - 1; j++) {
            if(indices[j] > indices[j + 1]) {
                int tmp2 = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp2;
            }
        }
    }

    return 1;
}

igo_dense* igo_compute_PAb_delta_sel(
    /* --- input --- */
    igo_sparse* PA_sel,
    igo_sparse* PA_sel_neg,
    igo_dense* b,
    igo_dense* b_staged_neg,
    int* sel_cols,
    int num_sel_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    int nrow = PA_sel->A->nrow;
    int ncol = PA_sel->A->ncol;

    igo_dense* PAb_delta = igo_zeros(nrow, 1, CHOLMOD_REAL, igo_cm);
    igo_dense* b_sel = igo_zeros(num_sel_cols, 1, CHOLMOD_REAL, igo_cm);
    igo_dense* b_sel_neg = igo_zeros(num_sel_cols, 1, CHOLMOD_REAL, igo_cm);

    double* bx = (double*) b->B->x;
    double* b_staged_neg_x = (double*) b_staged_neg->B->x;
    double* b_sel_x = (double*) b_sel->B->x;
    double* b_sel_neg_x = (double*) b_sel_neg->B->x;

    for(int idx = 0; idx < num_sel_cols; idx++) {
        int j = sel_cols[idx];

        b_sel_x[idx] = bx[j];
        b_sel_neg_x[idx] = b_staged_neg_x[j];
    }

    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};
    double alpha_negone[2] = {-1, -1};

    igo_sdmult(PA_sel, 0, alpha_one, alpha_zero, b_sel, PAb_delta, igo_cm);
    igo_sdmult(PA_sel_neg, 0, alpha_negone, alpha_one, b_sel_neg, PAb_delta, igo_cm);

    igo_free_dense(&b_sel, igo_cm);
    igo_free_dense(&b_sel_neg, igo_cm);

    return PAb_delta;
}

// Set the column nz of the selected columns 0
int igo_set_col_zero(
    /* --- inputs --- */
    int* col_indices,
    int len,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    for(int jidx = 0; jidx < len; jidx++) {
        int j = col_indices[jidx];
        if(A_staged_diff->data[j] > 0) {
            (*num_staged_cols)--;
            A_staged_diff->data[j] = 0;
        }
    }

    assert(*num_staged_cols >= 0);

    return 1;
}

// Set A_staged_diff = 0 of all columns after A_hat_col_start
int igo_set_A_hat_col_zero(
    /* --- inputs --- */
    int A_hat_col_start,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    for(int j = A_hat_col_start; j < A_staged_diff->len; j++) {
        assert(A_staged_diff->data[j] > 0);
        (*num_staged_cols)--;
        A_staged_diff->data[j] = 0;
    }

    assert(*num_staged_cols >= 0);

    return 1;
}

// Assume both inputs are vectors
// Do y += alpha Px 
int igo_add_sparse_to_dense(
    /* --- inputs --- */
    igo_sparse* x,
    double alpha,
    /* --- in/out --- */
    igo_dense* y,
    /* --- common --- */
    igo_common* igo_cm
) {
    assert(x->A->packed);
    int* xp = (int*) x->A->p;
    int* xi = (int*) x->A->i;
    double* xx = (double*) x->A->x;
    double* yx = (double*) y->B->x;
    for(int idx = 0; idx < xp[1]; idx++) {
        int i = xi[idx];
        yx[i] += alpha * xx[idx];
    }
    return 1;
}
