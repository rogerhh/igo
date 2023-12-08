#include "igo.h"

#include <assert.h>
#include <stdio.h>

int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_cm->cholmod_cm = malloc(sizeof(cholmod_common));
    cholmod_start(igo_cm->cholmod_cm);

    igo_cm->FACTOR_NCOL_ALLOC = 16;
    igo_cm->FACTOR_NZMAX_ALLOC = 32;
    igo_cm->FACTOR_DEFAULT_COL_SIZE = 16;
    igo_cm->DENSE_D_GROWTH = 16;

    igo_cm->A = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->b = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->L = igo_allocate_factor(0, 0, igo_cm);
    igo_cm->x = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->y = igo_allocate_dense(0, 0, 0, igo_cm);

    return 1;
}

int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_free_sparse(&(igo_cm->A), igo_cm);
    igo_free_dense(&(igo_cm->b), igo_cm);
    igo_free_factor(&(igo_cm->L), igo_cm);
    igo_free_dense(&(igo_cm->x), igo_cm);
    igo_free_dense(&(igo_cm->y), igo_cm);

    cholmod_finish(igo_cm->cholmod_cm);
    free(igo_cm->cholmod_cm);
    igo_cm->cholmod_cm = NULL;
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
    if(igo_cm == NULL) {
        return 0;
    }

    // Input checking
    int res = 0;

    // For a baseline implementation (Cholesky factorization without partial ordering)
    // 0. First resize factor L, this is to get the new variable ordering
    // 1. Concatenate A_tilde and A_hat into new_A (Still using the b_tilde variable), and b_tilde and b_hat into new_b (still using the b_tilde variable)
    // 2. Compute new_Ab = new_A * new_b
    // 3. Compute delta_Ab = P(new_Ab)
    // 4. igo_updown_solve(+1, new_A, L, y, delta_Ab)
    // 5. Exchange corresponding entries from igo_A and A_tilde, and igo_b and b_tilde
    // 6. Concatenate A_hat into igo_A, and b_hat into igo_b
    // 7. Compute old_ab = A_tilde * b_tilde
    // 8. Compute delta_Ab = -P(old_Ab)
    // 9. igo_updown_solve(-1, new_A, L, y, delta_Ab)
    // 10. triangular solve L.T x = y
    // 11. Release all memory allocated
    
    // 0. First resize factor L, this is to get the new variable ordering
    // Use the last col pointer and nz to figure out the last used entry in L->x, L->i
    igo_factor* igo_L = igo_cm->L;
    int* Lp = (int*) igo_L->L->p;
    int* Lprev = (int*) igo_L->L->prev;
    int* Lnz = (int*) igo_L->L->nz;
    int last_col = Lprev[igo_L->L->n];
    int cur_max_index = Lp[last_col] + Lnz[last_col];
    int new_x_len = A_hat->A->nrow;
    int old_x_len = igo_L->L->n;
    igo_resize_factor(new_x_len, igo_L->L->nzmax, igo_L, igo_cm);
    int* LPerm = (int*) igo_L->L->Perm;

    // 1. Concatenate A_tilde and A_hat into new_A, and b_tilde and b_hat into new_b
    igo_horzappend_sparse2(A_hat, A_tilde, igo_cm);
    igo_vertappend_sparse2(b_hat, b_tilde, igo_cm);

    // 2. Compute new_Ab = new_A * new_b
    igo_sparse* new_Ab = igo_ssmult(A_tilde, b_tilde, igo_cm);

    // 3. Compute delta_Ab = P(new_Ab)
    // First resize igo_cm->Ab
    igo_dense* delta_Ab = igo_allocate_dense(new_x_len, 1, new_x_len, igo_cm);
    double* delta_Ab_x = (double*) delta_Ab->B->x;
    int* new_Ab_i = (int*) new_Ab->A->i;
    double* new_Ab_x = (double*) new_Ab->A->x;
    int new_Ab_nnz = ((int*) new_Ab->A->p)[1];
    for(int i = 0; i < new_Ab_nnz; i++) {
        int new_Ab_row = new_Ab_i[i];
        int perm_Ab_row = LPerm[new_Ab_row];
        delta_Ab_x[perm_Ab_row] = new_Ab_x[i];
    }

    // 4. igo_updown(+1, new_A, L, y). We only need to run updown_solve once since we only need to update b once
    res = igo_updown_solve(true, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
    assert(res == 1);

    if(res != 1) { return 0; }
    
    // 5. Exchange corresponding entries from igo_A and A_tilde, and igo_b and b_tilde
    // When downdating, we don't need entries from A_hat, so we can remove it from 
    // A_tilde now. Use A_tilde's nrow for num of variables and A's ncol for num 
    // of factors
    res = igo_resize_sparse(A_tilde->A->nrow, igo_cm->A->A->ncol, A_tilde->A->nzmax,
                            A_tilde, igo_cm);
    res = igo_resize_sparse(igo_cm->b->B->nrow, 1, b_tilde->A->nzmax,
                            b_tilde, igo_cm);
    assert(res == 1);
    if(res != 1) { return 0; }

    assert(A_tilde->A->ncol == igo_cm->A->A->ncol);
    assert(A_tilde->A->packed);
    assert(igo_cm->A->A->packed);

    int* A_tilde_p = (int*) A_tilde->A->p;
    int* A_tilde_i = (int*) A_tilde->A->i;
    double* A_tilde_x = (double*) A_tilde->A->x;
    int* Ap = (int*) igo_cm->A->A->p;
    int* Ai = (int*) igo_cm->A->A->i;
    double* Ax = (double*) igo_cm->A->A->x;
    // Loop through all columns of A_tilde, here we assume A and A_tilde are packed
    for(int j = 0; j < A_tilde->A->ncol; j++) {
        int A_tilde_col_start = A_tilde_p[j];
        int A_tilde_col_end = A_tilde_p[j + 1];
        int A_col_start = Ap[j];
        int A_col_end = Ap[j + 1];
        int A_idx = A_col_start;
        for(int idx = 0; idx < A_tilde_col_end - A_tilde_col_start; idx++) {
            int A_tilde_idx = A_tilde_col_start + idx;
            int A_idx = A_col_start + idx;
            assert(A_tilde_i[A_tilde_idx] == Ai[A_idx]);

            double tmp_val = Ax[A_idx];
            Ax[A_idx] = A_tilde_x[A_tilde_idx];
            A_tilde_x[A_tilde_idx] = tmp_val;
        }
    }

    int bnz = ((int*) b_tilde->A->p)[1];
    int* b_tilde_i = (int*) b_tilde->A->i;
    double* b_tilde_x = (double*) b_tilde->A->x;
    double* bx = (double*) igo_cm->b->B->x;

    for(int i = 0; i < bnz; i++) {
        int brow = b_tilde_i[i];
        double tmp = bx[brow];
        bx[brow] = b_tilde_x[i];
        b_tilde_x[i] = tmp;
    }

    // 6. Concatenate A_hat into igo_A, and b_hat into igo_b
    igo_horzappend_sparse2(A_hat, igo_cm->A, igo_cm);
    igo_vertappend_sparse_to_dense2(b_hat, igo_cm->b, igo_cm);
    
    // 7. Compute old_ab = A_tilde * b_tilde
    igo_sparse* old_Ab = igo_ssmult(A_tilde, b_tilde, igo_cm);

    // 8. Compute delta_Ab = -P(old_Ab)
    int* old_Ab_i = (int*) old_Ab->A->i;
    double* old_Ab_x = (double*) old_Ab->A->x;
    int old_Ab_nnz = ((int*) old_Ab->A->p)[1];
    for(int i = 0; i < old_Ab_nnz; i++) {
        int old_Ab_row = old_Ab_i[i];
        int perm_Ab_row = LPerm[old_Ab_row];
        delta_Ab_x[perm_Ab_row] = -old_Ab_x[i];
    }

    // 9. igo_updown_solve(-1, new_A, L, y, delta_Ab)
    res = igo_updown_solve(false, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
    assert(res == 1);

    // 10. triangular solve DL.T x = y
    igo_dense* x_new = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
    igo_free_dense(&igo_cm->x, igo_cm);
    igo_cm->x = x_new;


    // 9. Release all memory allocated
    igo_free_sparse(&new_Ab, igo_cm);
    igo_free_sparse(&old_Ab, igo_cm);

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


void igo_print_cholmod_factor(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_factor* L,
    cholmod_common* cholmod_cm
) {
    cholmod_print_factor(L, name, cholmod_cm);

    bool is_ll_old = L->is_ll;

    // // Only print LL' factorization
    // cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, cholmod_cm);

    if(verbose >= 3) {
        printf("itype = %d, xtype = %d, dtype = %d\n", L->itype, L->xtype, L->dtype);
        printf("ordering = %d, is_ll = %d, is_super = %d, is_monotonic = %d\n", L->ordering, L->is_ll, L->is_super, L->is_monotonic);
        printf("nzmax = %d\n", L->nzmax);
    }

    if(verbose >= 2) {
        int* Lp = (int*) L->p;
        int* Li = (int*) L->i;
        int* Lnz = (int*) L->nz;
        double* Lx = (double*) L->x;
        int* LPerm = (int*) L->Perm;
        int* Lnext = (int*) L->next;
        int* Lprev = (int*) L->prev;
        int* LColCount = (int*) L->ColCount;

        printf("Lp = ");
        for(int j = 0; j < L->n + 1; j++) {
            printf("%d ", Lp[j]);
        }
        printf("\n");
        printf("Li = ");
        for(int j = 0; j < Lp[L->n]; j++) {
            printf("%d ", Li[j]);
        }
        printf("\n");
        printf("Lx = ");
        for(int j = 0; j < Lp[L->n]; j++) {
            printf("%f ", Lx[j]);
        }
        printf("\n");
        printf("nz = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", Lnz[j]);
        }
        printf("\n");
        printf("Next = ");
        for(int j = 0; j < L->n + 2; j++) {
            printf("%d ", Lnext[j]);
        }
        printf("\n");
        printf("Prev = ");
        for(int j = 0; j < L->n + 2; j++) {
            printf("%d ", Lprev[j]);
        }
        printf("\n");
        printf("Perm = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", LPerm[j]);
        }
        printf("\n");
    }

    if(verbose >= 1) {
        if(!L->is_super) {
            // Access the data arrays
            double* values = (double*)L->x;
            int* row_indices = (int*)L->i;
            int* column_pointers = (int*)L->p;
            int* nz = (int*) L->nz;

            // Iterate through the columns
            for (int j = 0; j < L->n; j++) {
                int start = column_pointers[j];
                int size = nz[j];

                // Iterate through the non-zero entries in the current column
                for (int i = start; i < start + size; i++) {

                    double value = values[i];
                    int row = row_indices[i];
                    // printf("row = %d \n", row);

                    printf("Value at (%d, %d) = %f\n", row, j, value);
                }
            }
        }
        else {
            int* Lsuper = (int*) L->super;
            int* Lpi = (int*) L->pi;
            int* Lpx = (int*) L->px;
            int* Ls = (int*) L->s;
            double* Lx = (double*) L->x;
            for(int js = 0; js < L->nsuper; js++) {
                int super_width = Lsuper[js + 1] - Lsuper[js];
                printf("Supernode: ");
                for(int j = 0; j < super_width; j++) {
                    printf("%d ", Lsuper[js] + j);
                }
                printf("\n");

                int super_height = Lpi[js + 1] - Lpi[js];
                for(int i = 0; i < super_height; i++) {
                    printf("Row %d: ", Ls[Lpi[js] + i]);

                    for(int j = 0; j < super_width; j++) {
                        int idx = Lpx[js] + i + j * super_height;
                        printf("%f ", Lx[idx]);
                    }

                    printf("\n");
                }
            }
        }
    }

    // cholmod_change_factor(CHOLMOD_REAL, is_ll_old, 0, 1, 1, L, cholmod_cm);
}
