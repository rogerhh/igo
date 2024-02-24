#include "igo.h"
#include <cstring>

extern "C" {
#include "cholmod.h"
}

#include <assert.h>
#include <stdio.h>
#include <float.h>

#define MIN(x, y) (x < y? x : y)
#define MAX(x, y) (x > y? x : y)


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

    igo_cm->solve_partial = IGO_SOLVE_DECIDE;
    igo_cm->partial_thresh = IGO_DEFAULT_PARTIAL_THRESH;

    igo_cm->subfactor_grow = IGO_FACTOR_DEFAULT_SUBFACTOR_GROW;

    igo_cm->A = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->A_staged_neg = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->A_staged_diff = igo_allocate_vector_double(igo_cm->A_staged_neg->ncol_alloc, igo_cm);
    igo_cm->b = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->b_staged_neg = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->L = igo_allocate_factor(0, 0, igo_cm);
    // igo_cm->Ab = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->x = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->y = igo_allocate_dense(0, 0, 0, igo_cm);

    igo_cm->reorder_counter = 0;
    igo_cm->num_staged_cols = 0;

    igo_cm->AT = igo_allocate_AT_pattern(IGO_SPARSE_DEFAULT_NCOL_ALLOC, igo_cm);

    return 1;
}

int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_free_sparse(&(igo_cm->A), igo_cm);
    igo_free_sparse(&(igo_cm->A_staged_neg), igo_cm);
    igo_free_vector_double(&(igo_cm->A_staged_diff), igo_cm);
    igo_free_dense(&(igo_cm->b), igo_cm);
    igo_free_dense(&(igo_cm->b_staged_neg), igo_cm);
    igo_free_factor(&(igo_cm->L), igo_cm);
    // igo_free_dense(&(igo_cm->Ab), igo_cm);
    igo_free_dense(&(igo_cm->x), igo_cm);
    igo_free_dense(&(igo_cm->y), igo_cm);

    igo_free_AT_pattern(&(igo_cm->AT), igo_cm);

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
        if(igo_cm->A_staged_diff->data[j] > 0) {
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

static int igo_build_affected_submatrix (
    /* --- input --- */
    igo_sparse* A,
    igo_dense* b,
    igo_sparse* A_staged_neg,
    igo_dense* b_staged_neg,
    int A_hat_col_start,        // The start of A_hat columns. We will deal with A_hat manually
    igo_vector_double* A_staged_diff,
    int num_affected_rows,
    int* affected_rows,
    int* row_map,
    int* L_map,
    int* L_map_inv,
    igo_AT_pattern* AT,
    /* --- output --- */
    igo_sparse** Asub_handle,
    igo_dense** bsub_handle,
    igo_sparse** Asub_staged_neg_handle,
    igo_dense** bsub_staged_neg_handle,
    int* num_staged_diff_sub,
    igo_vector_double* staged_diff_sub,
    int* num_affected_cols,
    int* affected_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    int ncol = A->A->ncol;
    int nrow = A->A->nrow;
    int* colmark = (int*) malloc(ncol * sizeof(int));

    *num_affected_cols = 0;

    assert(ncol >= A_hat_col_start);
    memset(colmark, 0, A_hat_col_start * sizeof(int));
    memset(colmark + A_hat_col_start, 1, (ncol - A_hat_col_start) * sizeof(int));

    int* Ap = (int*) A->A->p;
    int* Ai = (int*) A->A->i;
    double* Ax = (double*) A->A->x;
    int* A_staged_neg_p = (int*) A_staged_neg->A->p;
    int* A_staged_neg_i = (int*) A_staged_neg->A->i;
    double* A_staged_neg_x = (double*) A_staged_neg->A->x;
    double* b_staged_neg_x = (double*) b_staged_neg->B->x;

    for(int idx = 0; idx < num_affected_rows; idx++) {
        int row = affected_rows[idx];
        for(int jidx = 0; jidx < AT->len[row]; jidx++) {
            int col = AT->i[row][jidx];
            if(colmark[col] != 0) { continue; }

            colmark[col] = 1;
            affected_cols[*num_affected_cols] = col;
            (*num_affected_cols)++;
        }
    }

    // Manually add in A_hat columns to the back
    for(int j = A_hat_col_start; j < ncol; j++) {
        affected_cols[*num_affected_cols] = j;
        (*num_affected_cols)++;
    }

    printf("num_affected_cols: %d\n", *num_affected_cols);
    printf("affected_cols: ");
    for(int i = 0; i < *num_affected_cols; i++) {
        printf("%d ", affected_cols[i]);
    }
    printf("\n");

    igo_sparse* Asub = igo_allocate_sparse(num_affected_rows, 
                                           *num_affected_cols, 
                                           0, igo_cm);

    int* Asub_p = (int*) Asub->A->p;
    int* Asub_i = (int*) Asub->A->i;
    double* Asub_x = (double*) Asub->A->x;

    Asub_p[0] = 0;
    int nzmax = 0;
    for(int jidx = 0; jidx < *num_affected_cols; jidx++) {
        int col = affected_cols[jidx];
        int nz = Ap[col + 1] - Ap[col];
        igo_resize_sparse(num_affected_rows, *num_affected_cols, nzmax + nz, Asub, igo_cm);
        int p1 = Asub_p[jidx];
        for(int iidx = Ap[col]; iidx < Ap[col + 1]; iidx++) {
            int i = Ai[iidx];
            if(row_map[i] == -1) { continue; }

            Asub_i[p1] = row_map[i];
            Asub_x[p1] = Ax[iidx];
            p1++;
        }

        Asub_p[jidx + 1] = p1;
        nzmax += Asub_p[jidx + 1] - Asub_p[jidx];
    }

    int Cset[1] = {0};
    igo_dense* bsub = igo_dense_submatrix(b, affected_cols, *num_affected_cols, 
                                          Cset, 1, 
                                          igo_cm);

    printf("affected row ordering: ");
    for(int i = 0; i < num_affected_rows; i++) {
        printf("%d ", affected_rows[i]);
    }
    printf("\n");
    igo_print_sparse(3, "Asub", Asub, igo_cm);
    igo_print_dense(3, "bsub", bsub, igo_cm);

    // Manually construct Asub_staged_neg and bsub_staged_neg
    // Since it depends on A_staged_diff
    igo_sparse* Asub_staged_neg = igo_allocate_sparse(num_affected_rows, 
                                                      *num_affected_cols,
                                                      Asub->A->nzmax,
                                                      igo_cm);
    igo_dense* bsub_staged_neg = igo_zeros(*num_affected_cols, 1, 
                                           CHOLMOD_REAL, 
                                           igo_cm);

    int* Asub_staged_neg_p = (int*) Asub_staged_neg->A->p;
    int* Asub_staged_neg_i = (int*) Asub_staged_neg->A->i;
    double* Asub_staged_neg_x = (double*) Asub_staged_neg->A->x;
    double* bsub_staged_neg_x = (double*) bsub_staged_neg->B->x;

    *num_staged_diff_sub = 0;
    Asub_staged_neg_p[0] = 0;
    for(int jidx = 0; jidx < *num_affected_cols; jidx++) {
        int col = affected_cols[jidx];
        printf("A_staged_diff[%d] = %.4e\n", col, A_staged_diff->data[col]);
        if(A_staged_diff->data[col] > 0) {
            int p1 = Asub_staged_neg_p[jidx];
            for(int iidx = A_staged_neg_p[col]; iidx < A_staged_neg_p[col + 1]; iidx++) {
                int i = A_staged_neg_i[iidx];
                if(row_map[i] == -1) { continue; }

                Asub_staged_neg_i[p1] = row_map[i];
                Asub_staged_neg_x[p1] = A_staged_neg_x[iidx];
                p1++;
            }

            Asub_staged_neg_p[jidx + 1] = p1;
            bsub_staged_neg_x[jidx] = b_staged_neg_x[col];

            (*num_staged_diff_sub)++;
        }
        else {
            Asub_staged_neg_p[jidx + 1] = Asub_staged_neg_p[jidx];
        }
        staged_diff_sub->data[jidx] = A_staged_diff->data[col];
    }

    igo_print_sparse(3, "Asub_staged_neg", Asub_staged_neg, igo_cm);
    igo_print_dense(3, "bsub_staged_neg", bsub_staged_neg, igo_cm);

    free(colmark);

    *Asub_handle = Asub;
    *bsub_handle = bsub;
    *Asub_staged_neg_handle = Asub_staged_neg;
    *bsub_staged_neg_handle = bsub_staged_neg;

    return 1;
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
    igo_vector_double* A_staged_diff,
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

// Set the column nz of the selected columns 0
static int igo_set_all_col_zero(
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) {
    memset(A_staged_diff->data, 0, A_staged_diff->len * sizeof(double));
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
    igo_vector_double* A_staged_diff,
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

static int igo_permute_L21 (
    /* --- input --- */
    igo_factor* L22,
    igo_sparse* L21,
    int num_L21_cols,
    int* L21_cols,
    int num_affected_rows,
    int* affected_rows, // maps new A rows to old A rows
    int* row_map,       // maps old A rows to new A rows
    int* L_map,         // maps old L rows to new L rows
    int* L_map_inv,     // maps new L rows to old L rows
    /* --- in/out --- */
    igo_factor* L,
    /* --- common --- */
    igo_common* igo_cm

) {
    printf("L21_cols: ");
    for(int i = 0; i < num_L21_cols; i++) {
        printf("%d ", L21_cols[i]);
    }
    printf("\n");

    int n2 = L22->L->n;
    int n = L->L->n;

    assert(n2 == num_affected_rows);
    
    // All allocated memory
    igo_sparse* PL21 = NULL;
    int* L21_to_L22_Perm = (int*) malloc(n2 * sizeof(int));
    int* L_to_L22_Perm = (int*) malloc(n * sizeof(int));

    int* L22_p = (int*) L22->L->p;
    int* L22_i = (int*) L22->L->i;
    int* L22_nz = (int*) L22->L->nz;
    int* L22_Perm = (int*) L22->L->Perm;
    int* L22_IPerm = (int*) L22->L->IPerm;
    double* L22_x = (double*) L22->L->x;

    int* Lp = (int*) L->L->p;
    int* Li = (int*) L->L->i;
    int* Lnz = (int*) L->L->nz;
    int* LColCount = (int*) L->L->ColCount;
    int* LPerm = (int*) L->L->Perm;
    int* LIPerm = (int*) L->L->IPerm;
    int* Lnext = (int*) L->L->next;
    int* Lprev = (int*) L->L->prev;
    int Lnzmax = L->L->nzmax;
    double* Lx = (double*) L->L->x;

    // First construct a faster lookup from new L_rows to new old_L_rows
    for(int j = 0; j < num_affected_rows; j++) {
        int old_L22_row = j;
        int old_L_row = L_map_inv[old_L22_row];
        int A_row = LPerm[old_L_row];
        int A22_row = row_map[A_row];
        int new_L22_row = L22_IPerm[A22_row];
        int new_L_row = L_map[new_L22_row];
        L21_to_L22_Perm[old_L22_row] = new_L22_row;
        L_to_L22_Perm[old_L_row] = new_L_row;
    }

    PL21 = igo_submatrix(L21, L21_to_L22_Perm, n2, NULL, -1, true, true, igo_cm);

    int* PL21_p = (int*) PL21->A->p;
    double* PL21_x = (double*) PL21->A->x;

    for(int jidx = 0; jidx < num_L21_cols; jidx++) {
        int Lcol = L21_cols[jidx];
        int nz = Lnz[Lcol];
        int p1 = Lp[Lcol];
        int p2 = p1 + nz;
        int idx;
        for(int idx = p2 - 1; idx >= p1; idx--) {
            int old_L_row = Li[idx]; 
            if(L_map[old_L_row] == -1) { break; }
        }
        idx++;

        int PL21_p1 = PL21_p[jidx];

        // Everything including and after idx is in the affected L rows
        for(; idx < p2; idx++, PL21_p1++) {
            int old_L_row = Li[idx];
            int new_L_row = L_to_L22_Perm[old_L_row];
            Li[idx] = new_L_row;
            Lx[idx] = PL21_x[PL21_p1];
        }

    }
    
    // Clean up all allocated memory
    igo_free_sparse(&PL21, igo_cm);
    free(L21_to_L22_Perm);
    free(L_to_L22_Perm);

    printf("Done permute L21 cols\n");

    return 1;
}

/* Merge L22 into L and y2 into y. And merge L21 back into L for the new ordering
 * */
static int igo_merge_partial (
    /* --- input --- */
    igo_factor* L22,
    igo_dense* y2,
    igo_sparse* L21,
    int num_affected_rows,
    int* affected_rows, // maps new A rows to old A rows
    int* row_map,       // maps old A rows to new A rows
    int* L_map,         // maps old L rows to new L rows
    int* L_map_inv,     // maps new L rows to old L rows
    /* --- in/out --- */
    igo_factor* L,
    igo_dense* y,
    /* --- common --- */
    igo_common* igo_cm
) {
    int n2 = L22->L->n;
    int n = L->L->n;
    int grow0 = igo_cm->cholmod_cm->grow0;
    int grow1 = igo_cm->cholmod_cm->grow1;
    int grow2 = igo_cm->cholmod_cm->grow2;

    int* L22_p = (int*) L22->L->p;
    int* L22_i = (int*) L22->L->i;
    int* L22_nz = (int*) L22->L->nz;
    int* L22_Perm = (int*) L22->L->Perm;
    int* L22_IPerm = (int*) L22->L->IPerm;
    double* L22_x = (double*) L22->L->x;

    int* Lp = (int*) L->L->p;
    int* Li = (int*) L->L->i;
    int* Lnz = (int*) L->L->nz;
    int* LColCount = (int*) L->L->ColCount;
    int* LPerm = (int*) L->L->Perm;
    int* LIPerm = (int*) L->L->IPerm;
    int* Lnext = (int*) L->L->next;
    int* Lprev = (int*) L->L->prev;
    int Lnzmax = L->L->nzmax;
    double* Lx = (double*) L->L->x;

    double* yx = (double*) y->B->x;
    double* y2_x = (double*) y2->B->x;

    // All allocated variables

    // TODO: Make this faster and not iterate through all of n
    // First reset all old L column spaces. 
    // If an old affected column is followed by another affected column, 
    // the first column will take all the space. The space will be distributed later
    igo_print_factor(3, "L", L, igo_cm);
    int jidx = n;
    int next_Lp = Lp[n];
    int col_count = 0;
    while(col_count < num_affected_rows) {
        jidx = Lprev[jidx];
        
        if(L_map[jidx] == -1) { continue; }
        col_count++;

        assert(Lp[jidx] <= next_Lp);

        int prev_jidx = Lprev[jidx];

        if(prev_jidx == n + 1) { break; }

        printf("jidx = %d, prev_jidx = %d\n", jidx, prev_jidx);

        if(L_map[prev_jidx] != -1) {
            Lp[jidx] = next_Lp;
        }

        next_Lp = Lp[jidx];

    }

    printf("Lp: ");
    for(int i = 0; i <= n; i++) {
        printf("%d ", Lp[i]);
    }
    printf("\n");

    assert(col_count == num_affected_rows);

    // Iterate forward and check if there is enough space for data and slack
    // If yes, copy it in and move Lp[Lnext[jidx]]
    // If not, allocate new space in the back and set all relevant pointers
    // At this point, jidx should be the first column in i/x that is affected
    jidx = Lprev[jidx];     // Do this so we can move the jidx updating to the front
    int j22 = 0;
    while(Lnext[jidx] != n) {
        jidx = Lnext[jidx];

        if(L_map[jidx] == -1) { continue; }

        int next_jidx = Lnext[jidx];

        int needed = MAX((double) L22_nz[j22] * grow1 + grow2, igo_cm->FACTOR_DEFAULT_COL_SIZE);

        printf("needed: %d\n", needed);
        printf("Lp: ");
        for(int i = 0; i <= n; i++) {
            printf("%d ", Lp[i]);
        }
        printf("\n");

        printf("%d %d\n", next_jidx, jidx);

        int space = Lp[next_jidx] - Lp[jidx];

        assert(space >= 0);

        if(needed <= space) {
            // If enough space, allocate some space for next column if next column is not fixed
            if(L_map[next_jidx] != -1) {
                Lp[next_jidx] = Lp[jidx] + needed;
            }
        }
        else {
            // If not enough space, allocate to the back
            // First move next pointer to this pointers location if needed
            if(L_map[next_jidx] != -1) {
                Lp[next_jidx] = Lp[jidx];
            }

            // Allocate to back
            Lp[jidx] = Lp[n];
            Lp[n] += needed;

            // Now fix linked list. Remove current jidx first
            int prev_jidx = Lprev[jidx];
            Lnext[prev_jidx] = next_jidx;
            Lprev[next_jidx] = prev_jidx;

            // Add current jidx to back
            int last_jidx = Lprev[n];
            Lnext[last_jidx] = jidx;
            Lprev[jidx] = -1;
        }

        j22++;
    }

    // Allocate enough space in factor
    igo_resize_factor(n, Lp[n], L, igo_cm);

    // Now actually copy data in and fix Perm and IPerm
    for(int j22 = 0; j22 < num_affected_rows; j22++) {
        int nz = L22_nz[j22];
        int jidx = L_map[j22];

        int p = Lp[jidx];
        int p22 = L22_p[j22];

        Lnz[jidx] = nz;
        LColCount[jidx] = nz;
        int A_row = affected_rows[L22_Perm[j22]];
        LPerm[jidx] = A_row;
        LIPerm[A_row] = jidx;

        for(int i = 0; i < nz; i++) {
            Lx[p] = L22_x[p22];
            Li[p] = L_map_inv[L22_i[p22]];
            p++;
            p22++;
        }

        yx[jidx] = y2_x[j22];
    }

    igo_print_factor(3, "L after merge", L, igo_cm);
    igo_print_dense(3, "y after merge", y, igo_cm);

    return 1;
}

static int igo_solve_full_x (
    /* --- inputs --- */   
    int num_affected_rows,
    int* L_map_inv,         // Maps new_L_row to old_L_row
    igo_factor* L,
    igo_dense* x2,
    igo_dense* y,
    /* --- outputs --- */   
    igo_dense** x_handle, 
    /* --- common --- */
    igo_common* igo_cm
) {
    // All allocated memory
    int* nz_arr = (int*) malloc(num_affected_rows * sizeof(int));
    double* D = (double*) malloc(num_affected_rows * sizeof(double));
    double* y_arr = (double*) malloc(num_affected_rows * sizeof(double));

    int* Lp = (int*) L->L->p;
    int* Li = (int*) L->L->i;
    int* Lnz = (int*) L->L->nz;
    double* Lx = (double*) L->L->x;

    double* x2_x = (double*) x2->B->x;
    double* yx = (double*) y->B->x;

    // Set L22 in L to be I
    for(int jidx = 0; jidx < num_affected_rows; jidx++) {
        printf("here-1\n");
        int j = L_map_inv[jidx];
        int p1 = Lp[j];

        printf("here0\n");

        D[jidx] = Lx[p1];
        nz_arr[jidx] = Lnz[j];
        printf("here1\n");

        Lx[p1] = 1;
        Lnz[j] = 1;
        printf("here2\n");

        y_arr[jidx] = yx[j];
        yx[j] = x2_x[jidx];
    }

    igo_print_dense(3, "y", y, igo_cm);

    igo_free_dense(x_handle, igo_cm);
    *x_handle = igo_solve(CHOLMOD_DLt, L, y, igo_cm);
    
    // Restore L22
    for(int jidx = 0; jidx < num_affected_rows; jidx++) {
        int j = L_map_inv[jidx];
        int p1 = Lp[j];

        Lx[p1] = D[jidx];
        Lnz[j] = nz_arr[jidx];

        yx[j] = y_arr[jidx];
    }

    // Clean up all allocated memory
    free(nz_arr);
    free(D);
    free(y_arr);

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
    // 3. Decide if partial solution or full solution. A partial solution is needed when the number of affected variables is low. If partial solution. Go into the S route
    // 4. Decide if batch or incremental or PCG case based on number of columns changed
    //    Incremental case is covered by the PCG case
    // B0. Analyze and factorize to get LL^T = (P_L * A) (P_L * A)^T
    // B1. Compute PAb = A * b
    // B2. Solve Ly = Ab. Need to solve for y for future incremental updates
    // B3. Clean up allocated memory
    // B4. Solve DLtx = y and unpermute x
    // B5. Reset A_staged_neg
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
    // S0. Given a list of affected variables
    // S1. Build partial L22 of all the affected variables
    // S2. Build A_neg = L21 and b_neg. Which is columns can contain the affected variables
    // S3. Get submatrix A and b from columns involving the affected variables
    // S4. Get submatrix A_staged_neg and b_staged_neg from columns involving the affected variables
    // S5. Get the subvector of A_staged_diff
    // S6. Get the submtrix y2 of the affected variables
    // S7. Call solve_increment3
    // S8. Merge y2 back into y
    // S9. Merge L22 back into L
    // S10. Set L22 to I temporarily and solve for the full x and unpermute x
    // S11. Restore L22 
    
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
    igo_factor* L22 = NULL;
    igo_sparse* PA_neg = NULL;
    igo_dense* b_neg = NULL;
    igo_sparse* A_sub = NULL;
    igo_dense* b_sub = NULL;
    igo_sparse* A_sub_staged_neg = NULL;
    igo_dense* b_sub_staged_neg = NULL;
    igo_dense* x2 = NULL;
    igo_dense* y2 = NULL;
    igo_vector_double* staged_diff_sub = NULL;
    igo_pcg_context* cxt = NULL;
    int* L21_cols = NULL;
    int* sel_cols = NULL;
    int* affected_rows = NULL;
    int* row_map = NULL;
    int* L_map = NULL;
    int* L_map_inv = NULL;
    int* affected_cols = NULL;

    int A_tilde_nz_cols = igo_count_nz_cols(A_tilde, igo_cm);
    int A_hat_nz_cols = A_hat->A->ncol;
    int changed_cols = A_tilde_nz_cols + A_hat->A->ncol;
    int orig_cols = igo_cm->A->A->ncol;
    int new_cols = orig_cols + A_hat_nz_cols;
    int num_affected_rows = 0;
    int num_affected_cols = 0;
    int num_L21_cols = 0;

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
        igo_horzappend_sparse2_pattern(A_hat, igo_cm->A_staged_neg, igo_cm);
        igo_resize_dense(new_cols, 1, new_cols, igo_cm->b_staged_neg, igo_cm);
        igo_vector_double_multi_pushback(new_cols - orig_cols, DBL_MAX, igo_cm->A_staged_diff, igo_cm);

        igo_AT_append_A_hat(A_hat, igo_cm->AT, igo_cm);
    }

    // 3. Decide if full or partial solution 
    int solve_partial = igo_cm->solve_partial;
    if(igo_cm->solve_partial == IGO_SOLVE_PARTIAL_DECIDE) {
        affected_rows = (int*) malloc(h_hat * sizeof(int));
        row_map = (int*) malloc(h_hat * sizeof(int));
        L_map = (int*) malloc(h_hat * sizeof(int));
        L_map_inv = (int*) malloc(h_hat * sizeof(int));
        igo_get_affected_rows(A_hat, 
                              igo_cm->A_staged_neg, 
                              igo_cm->A_staged_diff, 
                              orig_cols, 
                              igo_cm->L,
                              &num_affected_rows,
                              affected_rows,
                              row_map,
                              L_map,
                              L_map_inv,
                              igo_cm);

        printf("num affected_rows: %d\n", num_affected_rows);
        printf("affected_rows: ");
        for(int i = 0; i < num_affected_rows; i++) {
            printf("%d ", affected_rows[i]);
        }
        printf("\n");
        printf("row_map: ");
        for(int i = 0; i < h_hat; i++) {
            printf("%d ", row_map[i]);
        }
        printf("\n");
        printf("L_map: ");
        for(int i = 0; i < h_hat; i++) {
            printf("%d ", L_map[i]);
        }
        printf("\n");
        printf("L_map_inv: ");
        for(int i = 0; i < num_affected_rows; i++) {
            printf("%d ", L_map_inv[i]);
        }
        printf("\n");

        if(num_affected_rows >= igo_cm->partial_thresh * h_hat) {
            solve_partial = IGO_SOLVE_PARTIAL_TRUE;
        }
        else {
            solve_partial = IGO_SOLVE_PARTIAL_FALSE;
        }
    }
    
    if(solve_partial == IGO_SOLVE_PARTIAL_FALSE) {
    printf("Solve full\n");

        
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
        igo_set_all_col_zero(igo_cm->A_staged_diff, 
                             &igo_cm->num_staged_cols, igo_cm);


        // B5. Clean up allocated memory
        // printf("Before B5\n");
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
        int num_staged_cols_old = igo_cm->num_staged_cols;

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

            printf("A_tilde_nzcol: %d\n", A_tilde_nz_cols);
            printf("A_staged_neg nzcol: %d\n", num_staged_cols_old);
            printf("Selected relin cols count: %d\n", num_sel_relin_cols);
            printf("num iter: %d\n", cxt->num_iter);

            igo_unpermute_rows_dense(igo_cm->x, (int*) igo_cm->L->L->Perm, igo_cm);

        }
    }

    }
    else {  // solve_partial == IGO_SOLVE_PARTIAL_FALSE
        printf("Solve partial\n");

        // S0. Resize L and y if needed
        if(h_hat > h_orig) {
            // igo_print_factor(3, "L before resize", igo_cm->L, igo_cm);
            igo_resize_factor(h_hat, igo_cm->L->L->nzmax, igo_cm->L, igo_cm);
            // igo_print_factor(3, "L after resize", igo_cm->L, igo_cm);
            igo_resize_dense(h_hat, 1, h_hat, igo_cm->y, igo_cm);
        }

        // S1. Build partial L22 of all the affected variables
        L22 = igo_subfactor(igo_cm->L, num_affected_rows, affected_rows, row_map, L_map, L_map_inv, igo_cm);

        igo_print_factor(3, "L22", L22, igo_cm);

        // S2. Build A_neg = L21 and b_neg. Which is columns can contain the affected variables
        PA_neg = igo_allocate_sparse(0, 0, 0, igo_cm);
        b_neg = igo_allocate_dense(0, 1, 0, igo_cm);
        igo_get_neg_factor(igo_cm->L, 
                           igo_cm->y, 
                           num_affected_rows, 
                           affected_rows,
                           row_map,
                           L_map, 
                           L_map_inv,
                           &num_L21_cols, 
                           L21_cols,
                           PA_neg,
                           b_neg,
                           igo_cm);

        igo_print_sparse(3, "PA_neg", PA_neg, igo_cm);
        igo_print_dense(3, "b_neg", b_neg, igo_cm);

        // S3. Get submatrix A and b from columns involving the affected variables
        // S4. Get submatrix A_staged_neg and b_staged_neg from columns involving the affected variables
        // S5. Get the subvector of A_staged_diff
        int num_staged_diff_sub = 0;
        staged_diff_sub = igo_allocate_vector_double(new_cols, igo_cm);
        affected_cols = (int*) malloc(new_cols * sizeof(int));
        igo_build_affected_submatrix(igo_cm->A, 
                                     igo_cm->b, 
                                     igo_cm->A_staged_neg,
                                     igo_cm->b_staged_neg,
                                     orig_cols,
                                     igo_cm->A_staged_diff,
                                     num_affected_rows,
                                     affected_rows,
                                     row_map,
                                     L_map,
                                     L_map_inv,
                                     igo_cm->AT,
                                     &A_sub,
                                     &b_sub,
                                     &A_sub_staged_neg,
                                     &b_sub_staged_neg,
                                     &num_staged_diff_sub,
                                     staged_diff_sub,
                                     &num_affected_cols,
                                     affected_cols,
                                     igo_cm);

        printf("num_staged_diff_sub: %d\n", num_staged_diff_sub);
        printf("staged_diff_sub: ");
        for(int i = 0; i < A_sub_staged_neg->A->ncol; i++) {
            printf("%.4e ", staged_diff_sub->data[i]);
        }
        printf("\n");

        // S6. Get the submtrix y2 of the affected variables
        int Cset[1] = {0};
        y2 = igo_dense_submatrix(igo_cm->y, L_map_inv, num_affected_rows, Cset, 1, igo_cm);
        igo_dense* x2 = NULL;

        igo_print_dense(3, "y2", y2, igo_cm);

        printf("A_hat_start_col: %d\n", orig_cols);


        int solve_type = IGO_SOLVE_BATCH;

        // S7. Call solve_increment3
        igo_solve_increment3(A_sub, 
                             b_sub,
                             PA_neg,
                             b_neg,
                             orig_cols,
                             solve_type,
                             A_sub_staged_neg,
                             b_sub_staged_neg,
                             &num_staged_diff_sub,
                             staged_diff_sub,
                             &L22,
                             &y2,
                             &x2,
                             igo_cm);


        // S8. Permute L21 with the new permutation
        igo_permute_L21(L22, PA_neg,
                        num_L21_cols,
                        L21_cols,
                        num_affected_rows,
                        affected_rows, // maps new A rows to old A rows
                        row_map,       // maps old A rows to new A rows
                        L_map,         // maps old L rows to new L rows
                        L_map_inv,     // maps new L rows to old L rows
                        igo_cm->L,
                        igo_cm);

        // S8. Merge y2 back into y
        // S9. Merge L22 back into L
        igo_merge_partial(L22, y2, PA_neg, 
                          num_affected_rows, 
                          affected_rows, 
                          row_map, 
                          L_map,
                          L_map_inv,
                          igo_cm->L,
                          igo_cm->y,
                          igo_cm);

        // S10. Set L22 to I temporarily and solve for the full x and unpermute x
        // S11. Restore L22 
        igo_print_factor(3, "L22", L22, igo_cm);
        igo_solve_full_x(num_affected_rows, L_map_inv, igo_cm->L, x2, igo_cm->y, &igo_cm->x, igo_cm);
        igo_unpermute_rows_dense(igo_cm->x, (int*) igo_cm->L->L->Perm, igo_cm);
        igo_print_dense(3, "full x", igo_cm->x, igo_cm);

        // S12. Unmark columns of A_staged_neg that are merged into L
        if(solve_type == IGO_SOLVE_BATCH) {
            igo_set_all_col_zero(igo_cm->A_staged_diff, &igo_cm->num_staged_cols, igo_cm);
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
    igo_free_factor(&L22, igo_cm);
    igo_free_sparse(&PA_neg, igo_cm);
    igo_free_dense(&b_neg, igo_cm);
    igo_free_sparse(&A_sub, igo_cm);
    igo_free_dense(&b_sub, igo_cm);
    igo_free_sparse(&A_sub_staged_neg, igo_cm);
    igo_free_dense(&b_sub_staged_neg, igo_cm);
    igo_free_dense(&x2, igo_cm);
    igo_free_dense(&y2, igo_cm);
    igo_free_vector_double(&staged_diff_sub, igo_cm);
    free(cxt);
    free(L21_cols);
    free(sel_cols);
    free(affected_rows);
    free(row_map);
    free(L_map);
    free(L_map_inv);
    free(affected_cols);
    cxt = NULL;
    sel_cols = NULL;

    return 1;
}

/* Solve a subproblem incrementally. 
 * The solved problem is defined as follows.
 * On input,
 * A includes columns of A_hat
 * Let S be the column selection matrix for the columns where A_staged_diff != 0. This 
 * includes columns of A_hat even though the A_staged_diff for those columns are 0
 * Let S' be the selection matrix for columns where A_staged_diff = 0
 * Then, 
 * LL^T = (AS')(AS')^T - (A_staged_neg S)(A_staged_neg S)^T - A_neg A_neg^T
 * Ly = AS' S'^T b - A_staged_neg S S^T b_staged_neg - A_neg A_neg^T b_neg
 *
 * On return,
 * LL^T = (AS')(AS')^T + A_hat A_hat^T - (A_staged_neg S)(A_staged_neg S)^T - A_neg A_neg^T
 * Ly = AS' S'^T b + A_hat b_hat - A_staged_neg S S^T b_staged_neg - A_neg A_neg^T b_neg
 * but for adjusted S and S' (i.e. the number of columns selected by S should decrease,
 * and S' now must include columns corresponding to A_hat)
 * (AA^T - A_neg A_neg^T) x = Ab - A_neg b_neg
 * Pass handles for parameters that may be re-allocated */
int igo_solve_increment3 (
    /* --- input --- */   
    igo_sparse* A,
    igo_dense* b,
    igo_sparse* PA_neg,
    igo_dense* b_neg,
    int A_hat_col_start,    // All columns after this column is part of A_hat
    int solve_type,
    /* --- output --- */   
    igo_sparse* A_staged_neg,
    igo_dense* b_staged_neg,
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    igo_factor** L_handle,
    igo_dense** y_handle,
    /* --- output --- */
    igo_dense** x_handle,
    /* --- common --- */
    igo_common* igo_cm
) {
    // All allocated memory
    igo_sparse* H = NULL;
    igo_sparse* H_pos = NULL;
    igo_sparse* H_neg = NULL;
    igo_sparse* A_neg = NULL;
    igo_dense* PAb = NULL;

    // Convenience variables
    igo_factor* L = *L_handle;
    igo_dense* y = *y_handle;
    igo_dense* x = *x_handle;
    int n = L->L->n;
    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};
    double alpha_negone[2] = {-1, -1};

    /* Batch */
    // B1. Compute H = AA^T - A_neg A_neg^T
    // B2. Compute LL^T = H
    // B3. Solve Ly = Ab - A_neg b_neg
    // B4. Solve L^T x = y
    // B5. Reset A_staged_diff

    if(solve_type == IGO_SOLVE_BATCH) {

        /* Batch */
        // B1. Unpermute A_neg = P^T PA_neg
        A_neg = igo_submatrix(PA_neg, (int*) L->L->IPerm, n, 
                              NULL, -1, true, true, 
                              igo_cm);

        igo_print_sparse(3, "A_neg", A_neg, igo_cm);

        // B1. Compute H = AA^T - A_neg A_neg^T
        H_pos = igo_aat(A, NULL, -1, CHOLMOD_REAL, igo_cm);
        H_pos->A->stype = -1;
        igo_print_sparse(3, "H_pos", H_pos, igo_cm);
        H_neg = igo_aat(A_neg, NULL, -1, CHOLMOD_REAL, igo_cm);
        H_neg->A->stype = -1;
        igo_print_sparse(3, "H_neg", H_neg, igo_cm);

        H = igo_add(H_pos, H_neg, alpha_one, alpha_negone, true, false, igo_cm);
        igo_print_sparse(3, "H", H, igo_cm);


        // B2. Compute LDL^T = H
        igo_free_factor(L_handle, igo_cm);
        L = igo_analyze_and_factorize(H, igo_cm);
        *L_handle = L;

        igo_print_factor(3, "L", L, igo_cm);

        // B3. Solve Ly = Ab - A_neg b_neg
        igo_print_sparse(1, "A before Ab", A, igo_cm);
        igo_print_dense(1, "b before Ab", b, igo_cm);
        igo_print_sparse(1, "A_neg before Ab", A_neg, igo_cm);
        igo_print_dense(1, "b_neg before Ab", b_neg, igo_cm);

        PAb = igo_zeros(n, 1, CHOLMOD_REAL, igo_cm);
        igo_sdmult(A, 0, alpha_one, alpha_zero, b, PAb, igo_cm);
        igo_sdmult(A_neg, 0, alpha_negone, alpha_one, b_neg, PAb, igo_cm);

        igo_permute_rows_dense(PAb, (int*) L->L->Perm, igo_cm);


        igo_free_dense(y_handle, igo_cm);
        y = igo_solve(CHOLMOD_L, L, PAb, igo_cm);
        *y_handle = y;

        igo_print_factor(3, "L", L, igo_cm);
        igo_print_dense(3, "PAb", PAb, igo_cm);
        igo_print_dense(3, "y", y, igo_cm);

        // B3. Solve DLtx = y. Dont unpermute x here
        assert(x == NULL);
        x = igo_solve(CHOLMOD_DLt, L, y, igo_cm);
        *x_handle = x;

        igo_print_dense(3, "x", x, igo_cm);

        // B5. Reset A_staged_diff
        igo_set_all_col_zero(A_staged_diff, num_staged_cols, igo_cm);

    }


    /* PCG */
    // P1. Pick k highest columns of the larges difference in A_staged_diff. Columns are in Ck
    // P2. Get the submatrices A_sel = A[:,Ck], A_sel_neg = A_staged_neg[:,Ck]
    //     b_sel = b[Ck], b_sel_neg = b[Ck]
    // P3. Permute PA_sel = P_L * A_sel, PA_sel_neg = P_L * A_sel_neg
    // P4. Compute PAb = PA_sel * b_sel, PAb_sel_neg = PA_sel_neg * b_sel_neg
    // P5. Compute PAb_delta = PAb_sel - PAb_sel_neg as a dense vector
    // P6. Call igo_updown2_solve(PA_sel, PA_sel_neg, L, y, PAb_delta)
    // P7. Permute PA_hat = P_L * A_hat
    // P8. Compute PAb_hat = PA_hat * b_hat
    // P10. Call igo_updown_solve(1, PA_hat, L, y, PAb_hat)
    // P11. Set A_staged_neg[:, Ck] = 0 by setting the nz of those columns 0
    // P12. A_staged_neg == 0, solve DLtx = y and unpermute x
    // P13. Else, solve PCGNE (AA^T - A_neg A_neg^T) x = Ab - A_neg b_neg

    // Clean up all allocated memory
    igo_free_sparse(&H, igo_cm);
    igo_free_sparse(&H_pos, igo_cm);
    igo_free_sparse(&H_neg, igo_cm);
    igo_free_sparse(&A_neg, igo_cm);
    igo_free_dense(&PAb, igo_cm);

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
