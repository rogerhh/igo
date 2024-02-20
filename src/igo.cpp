#include "igo.h"
#include <cstring>

extern "C" {
#include "cholmod.h"
}

#include <assert.h>
#include <stdio.h>


int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_cm->cholmod_cm = (cholmod_common*) malloc(sizeof(cholmod_common));
    cholmod_start(igo_cm->cholmod_cm);

    // // Use natural ordering for now. TODO: Change this later
    // igo_cm->cholmod_cm->nmethods = 1;
    // igo_cm->cholmod_cm->method[0].ordering = CHOLMOD_NATURAL;
    // igo_cm->cholmod_cm->postorder = false;
    // igo_cm->cholmod_cm->final_ll = false;
    // igo_cm->cholmod_cm->final_pack = false;
    // Turning cholmod supernodal off as it fails with factor allocation right now
    igo_cm->cholmod_cm->supernodal = CHOLMOD_SIMPLICIAL;  
    igo_cm->cholmod_cm->grow0 = 2;
    igo_cm->cholmod_cm->grow1 = 2;
    igo_cm->cholmod_cm->grow2 = 16;

    igo_cm->FACTOR_NCOL_ALLOC = 16;
    igo_cm->FACTOR_NZMAX_ALLOC = 32;
    igo_cm->FACTOR_DEFAULT_COL_SIZE = IGO_FACTOR_DEFAULT_COL_SIZE;
    igo_cm->DENSE_D_GROWTH = 16;
    igo_cm->BATCH_SOLVE_THRESH = IGO_DEFAULT_BATCH_SOLVE_THRESH;
    igo_cm->REORDER_PERIOD = IGO_REORDER_PERIOD;
    igo_cm->SEL_COLS_RATE = IGO_DEFAULT_SEL_COLS_RATE;
    igo_cm->MIN_SEL_COLS = IGO_DEFAULT_MIN_SEL_COLS;
    igo_cm->pcg_rtol = IGO_DEFAULT_PCG_RTOL;
    igo_cm->pcg_atol = IGO_DEFAULT_PCG_ATOL;
    igo_cm->solve_type = IGO_SOLVE_DECIDE;

    igo_cm->A = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->A_staged_neg = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->A_staged_diff = (double*) malloc(igo_cm->A_staged_neg->ncol_alloc * sizeof(double));
    memset(igo_cm->A_staged_diff, 0, igo_cm->A_staged_neg->ncol_alloc * sizeof(double));
    igo_cm->b = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->b_staged_neg = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->L = igo_allocate_factor(0, 0, igo_cm);
    // igo_cm->Ab = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->x = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->y = igo_allocate_dense(0, 0, 0, igo_cm);

    igo_cm->reorder_counter = 0;
    igo_cm->num_staged_cols = 0;

    return 1;
}

int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_free_sparse(&(igo_cm->A), igo_cm);
    igo_free_sparse(&(igo_cm->A_staged_neg), igo_cm);
    free(igo_cm->A_staged_diff);
    igo_cm->A_staged_diff = NULL;
    igo_free_dense(&(igo_cm->b), igo_cm);
    igo_free_dense(&(igo_cm->b_staged_neg), igo_cm);
    igo_free_factor(&(igo_cm->L), igo_cm);
    // igo_free_dense(&(igo_cm->Ab), igo_cm);
    igo_free_dense(&(igo_cm->x), igo_cm);
    igo_free_dense(&(igo_cm->y), igo_cm);

    cholmod_finish(igo_cm->cholmod_cm);
    free(igo_cm->cholmod_cm);
    igo_cm->cholmod_cm = NULL;

    return 1;
}

// Check solution of Ly = Atb
static int igo_check_state1 (
    igo_common* igo_cm
) {
    int m = igo_cm->A->A->nrow;
    int n = igo_cm->A->A->ncol;

    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};

    igo_sparse* A_restored = igo_copy_sparse(igo_cm->A, igo_cm);
    igo_dense* b_restored = igo_copy_dense(igo_cm->b, igo_cm);
    igo_dense* ATx = igo_allocate_dense(n, 1, n, igo_cm);
    igo_dense* AATx = igo_allocate_dense(m, 1, m, igo_cm);
    igo_dense* Ab = igo_allocate_dense(m, 1, m, igo_cm);

    igo_dense* x = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
    igo_unpermute_rows_dense(x, (int*) igo_cm->L->L->Perm, igo_cm);

    int* A_restored_p = (int*) A_restored->A->p;
    double* A_restored_x = (double*) A_restored->A->x;
    int* A_staged_neg_p = (int*) igo_cm->A_staged_neg->A->p;
    double* A_staged_neg_x = (double*) igo_cm->A_staged_neg->A->x;
    double* b_restored_x = (double*) b_restored->B->x;
    double* b_staged_neg_x = (double*) igo_cm->b_staged_neg->B->x;
    for(int j = 0; j < n; j++) {
        if(igo_cm->A_staged_diff[j] > 0) {
            int p1 = A_restored_p[j];
            int p2 = A_staged_neg_p[j];
            int nz = A_restored_p[j + 1] - p1;
            memcpy(A_restored_x + p1, A_staged_neg_x + p2, nz * sizeof(double));

            b_restored_x[j] = b_staged_neg_x[j];
        }
    }

    igo_sdmult(A_restored, 1, alpha_one, alpha_zero, x, ATx, igo_cm);
    igo_sdmult(A_restored, 0, alpha_one, alpha_zero, ATx, AATx, igo_cm);
    igo_sdmult(A_restored, 0, alpha_one, alpha_zero, b_restored, Ab, igo_cm);

    // igo_print_factor(3, "L", igo_cm->L, igo_cm);

    // igo_print_sparse(3, "A_restored", A_restored, igo_cm);

    double eps = 1e-6;
    if(!igo_dense_eq(AATx, Ab, eps, igo_cm)) {
        double* x1 = (double*) AATx->B->x;
        double* x2 = (double*) Ab->B->x;
        for(int i = 0; i < m; i++) {
            if(fabs(x1[i] - x2[i]) > eps) {
                printf("Entry at %d %f %f, diff = %f\n", i, x1[i], x2[i], fabs(x1[i] - x2[i]));
            }
        }
        fflush(stdout);
    }
    
        // igo_print_dense(3, "AATx", AATx, igo_cm);
        // igo_print_dense(3, "Ab", Ab, igo_cm);
        // fflush(stdout);

    assert(igo_dense_eq(AATx, Ab, 1e-6, igo_cm));

    igo_free_sparse(&A_restored, igo_cm);
    igo_free_dense(&b_restored, igo_cm);
    igo_free_dense(&ATx, igo_cm);
    igo_free_dense(&AATx, igo_cm);
    igo_free_dense(&Ab, igo_cm);

    igo_free_dense(&x, igo_cm);

    return 1;
}

static igo_dense* igo_compute_PAb_delta_sel(
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

// Comment: In the current implementation *_tilde and *_hat are not the difference
// to the current entries, but the new values
int igo_solve_increment (
    /* --- inputs --- */   
    igo_sparse* A_tilde, 
    igo_sparse* b_tilde,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- outputs --- */
    // igo_dense* x,
    /* --- common --- */
    igo_common* igo_cm
) {
//     if(igo_cm == NULL) {
//         return 0;
//     }
// 
//     // Input checking
//     int res = 0;
// 
//     // For a baseline implementation (Cholesky factorization without partial ordering)
//     // 0. First resize factor L, this is to get the new variable ordering
//     // 1. Concatenate A_tilde and A_hat into new_A (Still using the b_tilde variable), and b_tilde and b_hat into new_b (still using the b_tilde variable)
//     // 2. Compute new_Ab = new_A * new_b
//     // 3. Compute delta_Ab = P(new_Ab)
//     // 4. igo_updown_solve(+1, new_A, L, y, delta_Ab)
//     // 5. Exchange corresponding entries from igo_A and A_tilde, and igo_b and b_tilde
//     // 6. Concatenate A_hat into igo_A, and b_hat into igo_b
//     // 7. Compute old_ab = A_tilde * b_tilde
//     // 8. Compute delta_Ab = -P(old_Ab)
//     // 9. igo_updown_solve(-1, new_A, L, y, delta_Ab)
//     // 10. triangular solve L.T x = y
//     // 11. Release all memory allocated
//     
//     // 0. First resize factor L, this is to get the new variable ordering
//     // Use the last col pointer and nz to figure out the last used entry in L->x, L->i
//     igo_factor* igo_L = igo_cm->L;
//     int* Lp = (int*) igo_L->L->p;
//     int* Lprev = (int*) igo_L->L->prev;
//     int* Lnz = (int*) igo_L->L->nz;
//     int last_col = Lprev[igo_L->L->n];
//     int cur_max_index = Lp[last_col] + Lnz[last_col];
//     int new_x_len = A_hat->A->nrow;
//     int old_x_len = igo_L->L->n;
//     igo_resize_factor(new_x_len, igo_L->L->nzmax, igo_L, igo_cm);
//     int* LPerm = (int*) igo_L->L->Perm;
// 
//     // 1. Concatenate A_tilde and A_hat into new_A, and b_tilde and b_hat into new_b
//     igo_horzappend_sparse2(A_hat, A_tilde, igo_cm);
//     igo_vertappend_sparse2(b_hat, b_tilde, igo_cm);
// 
//     // 2. Compute new_Ab = new_A * new_b
//     igo_sparse* new_Ab = igo_ssmult(A_tilde, b_tilde, 0, true, true, igo_cm);
// 
//     // 3. Compute delta_Ab = P(new_Ab)
//     // First resize igo_cm->Ab
//     igo_dense* delta_Ab = igo_allocate_dense(new_x_len, 1, new_x_len, igo_cm);
//     double* delta_Ab_x = (double*) delta_Ab->B->x;
//     int* new_Ab_i = (int*) new_Ab->A->i;
//     double* new_Ab_x = (double*) new_Ab->A->x;
//     int new_Ab_nnz = ((int*) new_Ab->A->p)[1];
//     for(int i = 0; i < new_Ab_nnz; i++) {
//         int new_Ab_row = new_Ab_i[i];
//         int perm_Ab_row = LPerm[new_Ab_row];
//         delta_Ab_x[perm_Ab_row] = new_Ab_x[i];
//     }
// 
//     // 4. igo_updown(+1, new_A, L, y). We only need to run updown_solve once since we only need to update b once
//     res = igo_updown_solve(true, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
//     assert(res == 1);
// 
//     if(res != 1) { return 0; }
//     
//     // 5. Exchange corresponding entries from igo_A and A_tilde, and igo_b and b_tilde
//     // When downdating, we don't need entries from A_hat, so we can remove it from 
//     // A_tilde now. Use A_tilde's nrow for num of variables and A's ncol for num 
//     // of factors
//     res = igo_resize_sparse(A_tilde->A->nrow, igo_cm->A->A->ncol, A_tilde->A->nzmax,
//                             A_tilde, igo_cm);
//     res = igo_resize_sparse(igo_cm->b->B->nrow, 1, b_tilde->A->nzmax,
//                             b_tilde, igo_cm);
//     assert(res == 1);
//     if(res != 1) { return 0; }
// 
//     assert(A_tilde->A->ncol == igo_cm->A->A->ncol);
//     assert(A_tilde->A->packed);
//     assert(igo_cm->A->A->packed);
// 
//     int* A_tilde_p = (int*) A_tilde->A->p;
//     int* A_tilde_i = (int*) A_tilde->A->i;
//     double* A_tilde_x = (double*) A_tilde->A->x;
//     int* Ap = (int*) igo_cm->A->A->p;
//     int* Ai = (int*) igo_cm->A->A->i;
//     double* Ax = (double*) igo_cm->A->A->x;
//     // Loop through all columns of A_tilde, here we assume A and A_tilde are packed
//     for(int j = 0; j < A_tilde->A->ncol; j++) {
//         int A_tilde_col_start = A_tilde_p[j];
//         int A_tilde_col_end = A_tilde_p[j + 1];
//         int A_col_start = Ap[j];
//         int A_col_end = Ap[j + 1];
//         int A_idx = A_col_start;
//         for(int idx = 0; idx < A_tilde_col_end - A_tilde_col_start; idx++) {
//             int A_tilde_idx = A_tilde_col_start + idx;
//             int A_idx = A_col_start + idx;
//             assert(A_tilde_i[A_tilde_idx] == Ai[A_idx]);
// 
//             double tmp_val = Ax[A_idx];
//             Ax[A_idx] = A_tilde_x[A_tilde_idx];
//             A_tilde_x[A_tilde_idx] = tmp_val;
//         }
//     }
// 
//     int bnz = ((int*) b_tilde->A->p)[1];
//     int* b_tilde_i = (int*) b_tilde->A->i;
//     double* b_tilde_x = (double*) b_tilde->A->x;
//     double* bx = (double*) igo_cm->b->B->x;
// 
//     for(int i = 0; i < bnz; i++) {
//         int brow = b_tilde_i[i];
//         double tmp = bx[brow];
//         bx[brow] = b_tilde_x[i];
//         b_tilde_x[i] = tmp;
//     }
// 
//     // 6. Concatenate A_hat into igo_A, and b_hat into igo_b
//     igo_horzappend_sparse2(A_hat, igo_cm->A, igo_cm);
//     igo_vertappend_sparse_to_dense2(b_hat, igo_cm->b, igo_cm);
//     
//     // 7. Compute old_ab = A_tilde * b_tilde
//     igo_sparse* old_Ab = igo_ssmult(A_tilde, b_tilde, 0, true, false, igo_cm);
// 
//     // 8. Compute delta_Ab = -P(old_Ab)
//     int* old_Ab_i = (int*) old_Ab->A->i;
//     double* old_Ab_x = (double*) old_Ab->A->x;
//     int old_Ab_nnz = ((int*) old_Ab->A->p)[1];
//     for(int i = 0; i < old_Ab_nnz; i++) {
//         int old_Ab_row = old_Ab_i[i];
//         int perm_Ab_row = LPerm[old_Ab_row];
//         delta_Ab_x[perm_Ab_row] = -old_Ab_x[i];
//     }
// 
//     // 9. igo_updown_solve(-1, new_A, L, y, delta_Ab)
//     res = igo_updown_solve(false, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
//     assert(res == 1);
// 
//     // 10. triangular solve DL.T x = y
//     igo_dense* x_new = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
//     igo_free_dense(&igo_cm->x, igo_cm);
//     igo_cm->x = x_new;
// 
// 
//     // 9. Release all memory allocated
//     igo_free_sparse(&new_Ab, igo_cm);
//     igo_free_sparse(&old_Ab, igo_cm);
// 
    return 1;
}

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

// Go through each nonzero columns of A_tilde_neg and check corresponding columns of A_staged_neg
// If nz = 0, then copy the entries into A_staged_neg. 
static int igo_replace_staged(
    igo_sparse* A_tilde, 
    igo_sparse* A_tilde_neg, 
    igo_sparse* A_staged_neg, 
    igo_sparse* b_tilde,
    igo_sparse* b_tilde_neg,
    igo_dense* b_staged_neg,
    double* A_staged_diff,
    int* num_staged_cols,
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
        double staged = A_staged_diff[j];

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

        A_staged_diff[j] = A_diff > b_diff? A_diff : b_diff;

        count++;
    }

    return 1;
}

// Sort the list into 2 halves, the first m elements are smaller than pivot
// the last len - m elements are greater than or equal to pivot
// Co-sort the indices list as well
// Returns m
static int pivot_list(double* vals, int* indices, int len, double pivot) {
    int idx1 = 0, idx2 = len - 1;
    while(idx1 != idx2) {
        if(vals[idx1] < pivot) { idx1++; }
        else {
            double t1 = vals[idx1];
            vals[idx1] = vals[idx2];
            vals[idx2] = t1;
            int t2 = indices[idx1];
            indices[idx1] = indices[idx2];
            indices[idx2] = t2;
            idx2--;
        }
    }

    return idx1;
}

// Return the indices[i] of the maximum k vals[i] and put in ret_list
static int igo_max_k_elem(
    /* --- inputs --- */
    double* vals, 
    int* indices, 
    int len,
    int k,
    /* --- outputs --- */
    int* ret_list
) {
    assert(len >= k);
    int p1 = 0, p2 = len;     // p1, p2 splits the list into 3 sections.
                              // We will find the section that len - k lies in
    int target = len - k;
    int len1 = 0, len2 = len;
    while(p1 != target) {
        assert(p2 > p1);
        assert(target > p1);
        assert(target < p2);


        int pivot_idx = (p1 + p2) / 2;
        int p_offset = pivot_list(vals + p1, indices + p1, p2 - p1, vals[pivot_idx]);
        int p = p1 + p_offset;

        // Everything above p is greater than everything below p
        if(target < p) { 
            p2 = p; 
        }
        else {
            p1 = p;
        }
    }

    memcpy(ret_list, indices + target, k * sizeof(double));

    return 1;
}

// Assume both inputs are vectors
// Do y += alpha Px 
static int igo_add_sparse_to_dense(
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

// Set the column nz of the selected columns 0
static int igo_set_col_zero(
    /* --- inputs --- */
    int* col_indices,
    int len,
    /* --- in/out --- */
    igo_sparse* A_staged_neg,
    double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    igo_check_invariant_sparse(A_staged_neg, igo_cm);
    for(int jidx = 0; jidx < len; jidx++) {
        int j = col_indices[jidx];
        if(A_staged_diff[j] > 0) {
            (*num_staged_cols)--;
            A_staged_diff[j] = 0;
        }
    }
    return 1;
}

// Set the column nz of the selected columns 0
static int igo_set_all_col_zero(
    /* --- in/out --- */
    igo_sparse* A_staged_neg,
    double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    igo_check_invariant_sparse(A_staged_neg, igo_cm);
    for(int j= 0; j < A_staged_neg->A->ncol; j++) {
        A_staged_diff[j] = 0;
    }
    *num_staged_cols = 0;
    return 1;
}

// Return a list of indices to at most max_k columns with the highest 
// difference in A and A_staged_neg
// Assume indices is already allocated
// Only consider the first ncol columns. The other columns columns will guarantee to be picked
static int igo_pick_k_highest_diff(
    /* --- inputs --- */
    int max_k,
    int ncol,
    double* A_staged_diff,
    int num_staged_cols,
    /* --- outputs --- */
    int* k,
    int* indices,
    /* --- common --- */
    igo_common* igo_cm
) {

    *k = 0;

    if(max_k == 0) { return 1; }

    if(max_k >= num_staged_cols) {
        for(int i = 0; i < ncol; i++) {
            double diff = A_staged_diff[i];
            if(diff > 0) {
                indices[(*k)++] = i;
            }
        }
    }
    else {

        double* sel_diff = (double*) malloc(max_k * sizeof(double));

        // TODO: Make this algorithm more efficient
        for(int i = 0; i < ncol; i++) {
            double diff = A_staged_diff[i];
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

// Comment: In the current implementation *_tilde and *_hat are not the difference
// to the current entries, but the new values
int igo_solve_increment2 (
    /* --- inputs --- */   
    igo_sparse* A_tilde, 
    igo_sparse* b_tilde,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- outputs --- */
    // igo_dense* x,
    /* --- common --- */
    igo_common* igo_cm
) {
    if(igo_cm == NULL) {
        return 0;
    }

    // Input checking
    int res = 0;

    // An implementation that allows for reordering
    // LL^T = (P_L * A) * (P_L * A)^T
    // In the batch case, the inputs do not need to be permuted
    // In the incremental case, the inputs need to be permuted with P_L
    // 0. Resize A, A_tilde to be the same height as A_hat
    // 1. Replace entries in A with A_tilde and b with b_tilde to get A_tilde_neg and b_tilde_neg
    // 2. Append A with A_hat and b with b_hat. Append A_staged_neg with the pattern of A_hat
    // 3. Decide if batch or incremental or PCG case based on number of columns changed
    //    Incremental case is covered by the PCG case
    // B0. Analyze and factorize to get LL^T = (P_L * A) (P_L * A)^T
    // B1. Compute PAb = A * b
    // B2. Solve Ly = Ab. Need to solve for y for future incremental updates
    // B3. Clean up allocated memory
    // B4. Solve DLtx = y and unpermute x
    // B5. Reset A_staged_neg
    // I0. Resize L and y if needed
    // I1. Permute PA_tilde = P_L * A_tilde, PA_tilde_neg = P_L * A_tilde_neg
    // I2. Compute PAb_tilde = PA_tilde * b_tilde, PAb_tilde_neg = PA_tilde_neg * b_tilde_neg
    // I3. Compute PAb_delta = PAb_tilde - PAb_tilde_neg as a dense vector
    // I8.1. Clean up allocated memory
    // I4. Call igo_updown2_solve(PA_tilde, PA_tilde_neg, L, y, PAb_delta)
    // I5. Permute PA_hat = P_L * A_hat
    // I6. Compute PAb_hat = PA_hat * b_hat
    // I7. Call igo_updown_solve(1, PA_hat, L, y, PAb_hat)
    // I8.2. Clean up allocated memory
    // I9. Solve DLtx = y and unpermute x
    // P0. Resize L and y if needed
    // P1. Go through columns of A_tilde_neg, if corresponding column in A_staged_neg is 0, replace with column in A_tilde_neg
    // P2. Go through columns of A_staged_neg and compare with corresponding columns in A. Pick the k highest columns of the largest difference. The column indices are in Ck
    // P3. Get the submatrices A_sel = A[:,Ck], A_sel_neg = A_staged_neg[:,Ck]
    // P4. Permute PA_sel = P_L * A_sel, PA_sel_neg = P_L * A_sel_neg
    // P5. Compute PAb = PA_sel * b_sel, PAb_sel_neg = PA_sel_neg * b_sel_neg
    // P6. Compute PAb_delta = PAb_sel - PAb_sel as a dense vector
    // P7. Call igo_updown2_solve(PA_sel, PA_sel_neg, L, y, PAb_delta)
    // P8. Permute PA_hat = P_L * A_hat 
    // P9. Compute PAb_hat = A_hat * b_hat
    // P10. Call igo_updown_solve(1, PA_hat, L, y, PAb_hat)
    // P11. Set A_stages_neg[:, Ck] = 0 by setting the nz of those columns 0
    // P12. If A_staged_neg == 0, solve DLtx = y and unpermute x
    // P13. Else, solve PCGNE AA^Tx = Ab
    // P11. Clean up allocated memory
    
    // Convenience variables
    cholmod_common* cholmod_cm = igo_cm->cholmod_cm;
    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};
    double alpha_negone[2] = {-1, -1};

    // All allocated variables
    igo_sparse* A_tilde_neg = NULL;
    igo_sparse* b_tilde_neg = NULL;
    igo_sparse* PA_sel = NULL;
    igo_sparse* PA_sel_neg = NULL;
    igo_dense* PAb_delta = NULL;
    igo_sparse* PA = NULL;
    igo_sparse* PA_hat = NULL;
    igo_sparse* PAb_hat = NULL;
    igo_dense* PAb = NULL;
    igo_pcg_context* cxt = NULL;
    int* sel_cols = NULL;

    int A_tilde_nz_cols = igo_count_nz_cols(A_tilde, igo_cm);
    int A_hat_nz_cols = A_hat->A->ncol;
    int changed_cols = A_tilde_nz_cols + A_hat->A->ncol;
    int orig_cols = igo_cm->A->A->ncol;
    int new_cols = orig_cols + A_hat_nz_cols;

    int h_orig = igo_cm->A->A->nrow;
    int h_hat = A_hat->A->nrow;
    int h_tilde = A_tilde->A->nrow;

    assert(h_hat >= h_orig);
    assert(h_hat > 0);

    // 0. Resize A, A_tilde to be the same height as A_hat
    // We should support both the case where height of A_hat == height of A_tilde 
    // and the case where height of A_hat >= height of A_tilde
    // printf("Before 0\n");
    if(h_tilde < h_hat) {
        igo_resize_sparse(h_hat, A_tilde->A->ncol, A_tilde->A->nzmax, A_tilde, igo_cm);
    }
    if(h_orig < h_hat) {
        igo_resize_sparse(h_hat, igo_cm->A->A->ncol, igo_cm->A->A->nzmax, igo_cm->A, igo_cm);
    }
    
    // 1. Replace entries in A with A_tilde and b with b_tilde to get A_tilde_neg and b_tilde_neg
    // printf("Before 1\n");
    if(A_tilde_nz_cols > 0) {
        A_tilde_neg = igo_replace_sparse(igo_cm->A, A_tilde, igo_cm);
        b_tilde_neg = igo_replace_dense(igo_cm->b, b_tilde, igo_cm);
    }
    
    // 2. Append A with A_hat and b with b_hat. Append A_staged_neg with the pattern of A_hat
    //    and append b_staged_neg with the pattern of b_hat
    // printf("Before 2\n");
    if(A_hat_nz_cols > 0) {
        igo_horzappend_sparse2(A_hat, igo_cm->A, igo_cm);
        igo_vertappend_sparse_to_dense2(b_hat, igo_cm->b, igo_cm);
        
        // FIXME: Make this cleaner
        int old_ncol_alloc = igo_cm->A_staged_neg->ncol_alloc;
        igo_horzappend_sparse2_pattern(A_hat, igo_cm->A_staged_neg, igo_cm);
        igo_resize_dense(new_cols, 1, new_cols, igo_cm->b_staged_neg, igo_cm);
        int new_ncol_alloc = igo_cm->A_staged_neg->ncol_alloc;
        if(new_ncol_alloc > old_ncol_alloc) {
            igo_cm->A_staged_diff = 
                (double*) realloc(igo_cm->A_staged_diff, new_ncol_alloc * sizeof(double));
            memset(igo_cm->A_staged_diff + old_ncol_alloc, 0, 
                    (new_ncol_alloc - old_ncol_alloc) * sizeof(double));
        }
    }
        
    // 3. Decide if batch or incremental case based on number of columns changed
    // printf("Before 3\n");
    int solve_type = igo_cm->solve_type;
    if(igo_cm->solve_type == IGO_SOLVE_DECIDE) {
        if(changed_cols > orig_cols * igo_cm->BATCH_SOLVE_THRESH) {
            solve_type = IGO_SOLVE_BATCH;
        }
        else if(++igo_cm->reorder_counter >= igo_cm->REORDER_PERIOD) {
            solve_type = IGO_SOLVE_BATCH;
        }
        else if(changed_cols > 0.5 * igo_cm->L->L->n) {
            solve_type = IGO_SOLVE_BATCH;
        }
        else {
            solve_type = IGO_SOLVE_PCG;
        }
    }
    
    if(solve_type == IGO_SOLVE_BATCH) {
        // printf("Batch\n");
        igo_cm->reorder_counter = 0;

        igo_free_factor(&igo_cm->L, igo_cm);
        igo_free_dense(&igo_cm->y, igo_cm);
        igo_free_dense(&igo_cm->x, igo_cm);

        // B0. Analyze and factorize to get LL^T = (P_L * A) (P_L * A)^T
        // printf("Before B0\n");
        igo_cm->L = igo_analyze_and_factorize(igo_cm->A, igo_cm);
        
        // B1. Compute Ab = A * b
        // printf("Before B1\n");
        double alpha[2] = {1, 1};
        double beta[2] = {0, 0};
        PAb = igo_zeros(h_hat, 1, CHOLMOD_REAL, igo_cm);
        igo_sdmult(igo_cm->A, 0, alpha, beta, igo_cm->b, PAb, igo_cm);
        igo_permute_rows_dense(PAb, (int*) igo_cm->L->L->Perm, igo_cm);

        // B2. Solve Ly = Ab. Need to solve for y for future incremental updates
        // printf("Before B2\n");
        igo_cm->y = igo_solve(CHOLMOD_L, igo_cm->L, PAb, igo_cm);

        // B3. Solve DLtx = y and unpermute x
        // printf("Before B3\n");
        igo_cm->x = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
        igo_unpermute_rows_dense(igo_cm->x, (int*) igo_cm->L->L->Perm, igo_cm);

        // B4. Reset A_staged_neg
        // printf("Before B4\n");
        igo_set_all_col_zero(igo_cm->A_staged_neg, 
                             igo_cm->A_staged_diff, 
                             &igo_cm->num_staged_cols, igo_cm);


        // B5. Clean up allocated memory
        // printf("Before B5\n");
    }
    else if(solve_type == IGO_SOLVE_INCREMENTAL) {
        // printf("Incremental\n");
        // I0. Resize L and y if needed
        if(h_hat > h_orig) {
            // igo_print_factor(3, "L before resize", igo_cm->L, igo_cm);
            igo_resize_factor(h_hat, igo_cm->L->L->nzmax, igo_cm->L, igo_cm);
            // igo_print_factor(3, "L after resize", igo_cm->L, igo_cm);
            igo_resize_dense(h_hat, 1, h_hat, igo_cm->y, igo_cm);
        }

        if(A_tilde_nz_cols > 0) {

        // I1. Permute PA_tilde = P_L * A_tilde, PA_tilde_neg = P_L * A_tilde_neg
        int* P = (int*) igo_cm->L->L->Perm;
        igo_sparse* PA_tilde = igo_submatrix(A_tilde, P, h_hat,
                                             NULL, -1, true, true, igo_cm);
        igo_sparse* PA_tilde_neg = igo_submatrix(A_tilde_neg, P, h_hat,
                                                 NULL, -1, true, true, igo_cm);

        // I2. Compute PAb_tilde = PA_tilde * b_tilde, PAb_tilde_neg = PA_tile_neg * b_tilde_neg
        igo_sparse* PAb_tilde = igo_ssmult(PA_tilde, b_tilde, 
                                           0, true, true, 
                                           igo_cm);
        igo_sparse* PAb_tilde_neg = igo_ssmult(PA_tilde_neg, b_tilde_neg, 
                                               0, true, true, 
                                               igo_cm);

        // I3. Compute PAb_delta = PAb_tilde - PAb_tilde_neg as a dense vector
        int PAb_tilde_nz = ((int*) PAb_tilde->A->p)[1];
        int* PAb_tilde_i = (int*) PAb_tilde->A->i;
        double* PAb_tilde_x = (double*) PAb_tilde->A->x;
        double* PAb_tilde_neg_x = (double*) PAb_tilde_neg->A->x;

        igo_dense* PAb_delta = igo_zeros(h_hat, 1, CHOLMOD_REAL, igo_cm);
        double* PAb_delta_x = (double*) PAb_delta->B->x;

        for(int i = 0; i < PAb_tilde_nz; i++) {
            int PAb_row = PAb_tilde_i[i];
            PAb_delta_x[PAb_row] = PAb_tilde_x[i] - PAb_tilde_neg_x[i];
        }

        // I4. Call igo_updown2_solve(PA_tilde, PA_tilde_neg, L, y, PAb_delta)
        igo_drop_cols_sparse(PA_tilde, igo_cm);
        igo_drop_cols_sparse(PA_tilde_neg, igo_cm);
        igo_updown2_solve(PA_tilde, PA_tilde_neg, igo_cm->L, igo_cm->y, PAb_delta, igo_cm);


        // I8.1. Clean up allocated memory
        igo_free_sparse(&PA_tilde, igo_cm);
        igo_free_sparse(&PA_tilde_neg, igo_cm);
        igo_free_sparse(&PAb_tilde, igo_cm);
        igo_free_sparse(&PAb_tilde_neg, igo_cm);
        igo_free_dense(&PAb_delta, igo_cm);

        }

        if(A_hat_nz_cols > 0) {

        // I5. Permute PA_hat = P_L * A_hat
        int* P = (int*) igo_cm->L->L->Perm;
        igo_sparse* PA_hat = igo_submatrix(A_hat, P, h_hat,
                                           NULL, -1, true, true, igo_cm);

        // I6. Compute PAb_hat = PA_hat * b_hat
        igo_sparse* PAb_hat = igo_ssmult(PA_hat, b_hat, 
                                         0, true, true, 
                                         igo_cm);
        
        igo_dense* PAb_delta = igo_zeros(h_hat, 1, CHOLMOD_REAL, igo_cm);
        int PAb_hat_nz = ((int*) PAb_hat->A->p)[1];
        int* PAb_hat_i = (int*) PAb_hat->A->i;
        double* PAb_hat_x = (double*) PAb_hat->A->x;
        double* PAb_delta_x = (double*) PAb_delta->B->x;

        for(int i = 0; i < PAb_hat_nz; i++) {
            int brow = PAb_hat_i[i];
            PAb_delta_x[brow] = PAb_hat_x[i];
        }

        // I7. Call igo_updown_solve(1, PA_hat, L, y, PAb_hat)
        igo_updown_solve(1, PA_hat, igo_cm->L, igo_cm->y, PAb_delta, igo_cm);

        // I8.2. Clean up allocated memory
        igo_free_sparse(&PA_hat, igo_cm);
        igo_free_sparse(&PAb_hat, igo_cm);
        igo_free_dense(&PAb_delta, igo_cm);

        }

    }
   else if(solve_type == IGO_SOLVE_PCG) {
        // printf("PCG\n");
        
        // P0. Resize L and y if needed
        // printf("Before P0\n");

        if(h_hat > h_orig) {
            // igo_print_factor(3, "L before resize", igo_cm->L, igo_cm);
            igo_resize_factor(h_hat, igo_cm->L->L->nzmax, igo_cm->L, igo_cm);
            // igo_print_factor(3, "L after resize", igo_cm->L, igo_cm);
            igo_resize_dense(h_hat, 1, h_hat, igo_cm->y, igo_cm);
        }

        // P1. Go through columns of A_tilde_neg, if corresponding column in A_staged_neg is 0, replace with column in A_tilde_neg
        // printf("Before P1\n");
        igo_replace_staged(A_tilde, A_tilde_neg, igo_cm->A_staged_neg, 
                           b_tilde, b_tilde_neg, igo_cm->b_staged_neg,
                           igo_cm->A_staged_diff, &igo_cm->num_staged_cols,
                           igo_cm);

        // P2. Go through columns of A_staged_neg and compare with corresponding columns in A. Pick the k highest columns of the largest difference. The column indices are in Ck
        // printf("Before P2\n");
        int percent_sel_cols = ((double) h_hat) * igo_cm->SEL_COLS_RATE;
        int max_num_sel_cols = percent_sel_cols > igo_cm->MIN_SEL_COLS? 
                                percent_sel_cols : igo_cm->MIN_SEL_COLS;
        int num_sel_cols = 0;
        sel_cols = (int*) malloc((max_num_sel_cols + A_hat_nz_cols) * sizeof(int));
        igo_pick_k_highest_diff(max_num_sel_cols, orig_cols, 
                                igo_cm->A_staged_diff,
                                igo_cm->num_staged_cols,
                                &num_sel_cols, sel_cols, 
                                igo_cm);

        int num_sel_relin_cols = num_sel_cols;

        // printf("num_sel_cols2 = %d\n", num_sel_cols);
        // printf("selected cols %p: ", sel_cols);
        // for(int i = 0; i < num_sel_cols; i++) {
        //     printf("%d ", sel_cols[i]);
        // }
        // printf("\n");

        // P3. Get the submatrices A_sel = A[:,Ck], A_sel_neg = A_staged_neg[:,Ck]
        // P4. Permute PA_sel = P_L * A_sel, PA_sel_neg = P_L * A_sel_neg
        int* P = (int*) igo_cm->L->L->Perm;
        PA_sel = igo_submatrix(igo_cm->A, 
                               P, h_hat,
                               sel_cols, num_sel_cols, 
                               true, true, igo_cm);

        PA_sel_neg = igo_submatrix(igo_cm->A_staged_neg, 
                                   P, h_hat,
                                   sel_cols, num_sel_cols, 
                                   true, true, igo_cm);
        
        // P5. Compute Ab_delta = A_tilde * b_tilde - A_tilde_neg * b_tilde + A_hat * b_hat
        // Then permute it to get PAb_delta
        // printf("Before P5\n");
        
        PAb_delta = igo_compute_PAb_delta_sel(PA_sel, PA_sel_neg, 
                                              igo_cm->b, igo_cm->b_staged_neg,
                                              sel_cols, num_sel_cols,
                                              igo_cm);


        // P6. Call igo_updown2_solve(PA_sel, PA_sel_neg, L, y, PAb_delta)
        // printf("Before P6\n");
        igo_updown2_solve(PA_sel, PA_sel_neg, igo_cm->L, igo_cm->y, PAb_delta, igo_cm);

        // P7. Set A_stages_neg[:, Ck] = 0 by setting the nz of those columns 0
        // printf("Before P7\n");
        igo_set_col_zero(sel_cols, num_sel_cols, 
                         igo_cm->A_staged_neg, 
                         igo_cm->A_staged_diff, 
                         &igo_cm->num_staged_cols,
                         igo_cm);


        // P8. Do a single pass on Ahat columns
        PA_hat = igo_submatrix(A_hat, P, h_hat, NULL, -1, true, true, igo_cm);
        PAb_hat = igo_ssmult(PA_hat, b_hat, 0, true, true, igo_cm);
        igo_add_sparse_to_dense(PAb_hat, 1, PAb_delta, igo_cm);
        igo_updown_solve(1, PA_hat, igo_cm->L, igo_cm->y, PAb_delta, igo_cm);

        // igo_check_state1(igo_cm);

        // P8. If A_staged_neg == 0
        // printf("Before P8\n");

        if(igo_cm->num_staged_cols == 0) {
            // P9. If num_staged_nzcol == 0, solve DLtx = y and unpermute x
            // printf("Before P9\n");
            igo_free_dense(&igo_cm->x, igo_cm);
            igo_cm->x = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
            igo_unpermute_rows_dense(igo_cm->x, (int*) igo_cm->L->L->Perm, igo_cm);
        }
        else {
            // P10. Else, first get initial guess. DLt Px = y
            // printf("Before P10\n");
            igo_free_dense(&igo_cm->x, igo_cm);
            igo_cm->x = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);

            // P11. solve PCGNE PAA^TP^TPx = PAb
            // printf("Before P11\n");

            double alpha_one[2] = {1, 1};
            double alpha_zero[2] = {0, 0};
            PA = igo_submatrix(igo_cm->A, P, h_hat, NULL, -1, true, true, igo_cm);
            PAb = igo_zeros(h_hat, 1, CHOLMOD_REAL, igo_cm);
            igo_sdmult(PA, 0, alpha_one, alpha_zero, igo_cm->b, PAb, igo_cm);

            cxt = (igo_pcg_context*) malloc(sizeof(igo_pcg_context));
            igo_solve_pcgne(PA, NULL, PAb, igo_cm->L, 
                            igo_cm->pcg_rtol, igo_cm->pcg_atol, h_hat, 
                            igo_cm->x, cxt, 
                            igo_cm);

            printf("A_staged_neg nzcol: %d\n", igo_cm->num_staged_cols);
            printf("Selected relin cols count: %d\n", num_sel_relin_cols);
            printf("num iter: %d\n", cxt->num_iter);

            igo_unpermute_rows_dense(igo_cm->x, (int*) igo_cm->L->L->Perm, igo_cm);

        }
    }

    // 4. Clean up allocated memory
    igo_free_sparse(&A_tilde_neg, igo_cm);
    igo_free_sparse(&b_tilde_neg, igo_cm);
    igo_free_sparse(&PA_sel, igo_cm);
    igo_free_sparse(&PA_sel_neg, igo_cm);
    igo_free_dense(&PAb_delta, igo_cm);
    igo_free_sparse(&PA, igo_cm);
    igo_free_sparse(&PA_hat, igo_cm);
    igo_free_sparse(&PAb_hat, igo_cm);
    igo_free_dense(&PAb, igo_cm);
    free(cxt);
    free(sel_cols);
    cxt = NULL;
    sel_cols = NULL;

    return 1;
}

/* 
Returns A = [A B]. This is needed as cholmod_horzcat copies the input matrices
*/
int igo_horzappend (
    /* --- input --- */  
    cholmod_sparse* B,
    /* --- in/out --- */
    cholmod_sparse* A
) {

}

/*
Trivially add rows to a sparse matrix
*/
int igo_sparse_addrows(
    /* --- input --- */
    int nrow,       /* Number of total rows to add */
    /* --- in/out --- */
    cholmod_sparse* A
) {
    A->nrow = nrow;
    return 1;
}

void igo_print_cholmod_dense(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_dense* B,
    cholmod_common* cholmod_cm
) {
    
    cholmod_print_dense(B, name, cholmod_cm);

    if(!verbose) {
        return;
    }

    if(verbose >= 1) {
        double* Bx = (double*) B->x;
        for(int i = 0; i < B->nrow; i++) {
            for(int j = 0; j < B->ncol; j++) {
                printf("%.17g ", Bx[j * B->d + i]);
            }
            printf("\n");
        }
    }
}
