#include "igo.h"

#include <assert.h>

int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_cm->cholmod_cm = malloc(sizeof(cholmod_common));
    cholmod_start(igo_cm->cholmod_cm);

    igo_cm->A = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->Ab = igo_allocate_dense(0, 0, 0, igo_cm);
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
    igo_free_dense(&(igo_cm->Ab), igo_cm);
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
    // 1. Concatenate A_tilde and A_hat into new_A, and b_tilde and b_hat into new_b
    // 2. Compute new_Ab = new_A * new_b
    // 3. Allocate delta_Ab as a 0 vector. Set delta_Ab to corresponding entries in new_Ab (This needs to be permuted)
    // 4. igo_updown_solve(+1, new_A, L, y, delta_Ab)
    // 5. delta_Ab should be 0 at this point. Set corresponding entries in Ab to entries in new_Ab, but move the old entries to permuted entries in delta_Ab
    // 6. Exchange corresponding entries from A and new_A
    // 7. igo_updown_solve(-1, new_A, L, y, delta_Ab)
    // 8. triangular solve L.T x = y
    
    // 0. First resize factor L, this is to get the new variable ordering
    igo_factor* igo_L = igo_cm->L;
    igo_print_factor(3, "igo_L before resize", igo_L, igo_cm);
    int new_x_len = b_hat->A->nrow;
    int old_x_len = igo_L->L->n;
    printf("%d %d %d\n", new_x_len, old_x_len, igo_L->L->nzmax);
    igo_resize_factor(new_x_len, igo_L->L->nzmax + new_x_len - old_x_len, igo_L, igo_cm);
    int* LPerm = (int*) igo_L->L->Perm;

    igo_print_factor(3, "igo_L after resize", igo_L, igo_cm);

    // 1. Concatenate A_tilde and A_hat into new_A, and b_tilde and b_hat into new_b
    igo_horzappend_sparse2(A_hat, A_tilde, igo_cm);
    igo_vertappend_sparse2(b_hat, b_tilde, igo_cm);

    // 2. Compute new_Ab = new_A * new_b
    igo_sparse* new_Ab = igo_ssmult(A_tilde, b_tilde, igo_cm);

    // 3. Allocate delta_Ab as a 0 vector. Set delta_Ab to corresponding entries in new_Ab (This must be permuted)
    igo_dense* delta_Ab = igo_allocate_dense(new_x_len, 1, new_x_len, igo_cm);
    int* Abi = (int*) new_Ab->A->i;
    double* new_Abx = (double*) new_Ab->A->x;
    double* delta_Abx = (double*) delta_Ab->B->x;
    for(int i = 0; i < new_Ab->A->nzmax; i++) {
        int Ab_row = Abi[i];
        int perm_Ab_row = LPerm[Ab_row];
        delta_Abx[perm_Ab_row] = new_Abx[i];
    }

    // 4. igo_updown_solve(+1, new_A, L, y, delta_Ab)
    res = igo_updown_solve(1, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
        printf("after updown solve\n");
        fflush(stdout);
    assert(res == 1);

    if(res != 1) { return 0; }
    
    // 5. delta_Ab should be 0 at this point. Set corresponding entries in Ab to entries in new_Ab, but move the old entries to permuted entries in delta_Ab
    double* Abx = (double*) igo_cm->Ab->B->x;
    for(int i = 0; i < new_Ab->A->nzmax; i++) {
        int Ab_row = Abi[i];
        int perm_Ab_row = LPerm[Ab_row];
        delta_Abx[perm_Ab_row] = Abx[i];
        Abx[i] = new_Abx[i];
    }
    
    // 6. Exchange corresponding entries from A and A_tilde
    // When downdating, we don't need entries from A_hat, so we can remove it from 
    // A_tilde now
    res = igo_resize_sparse(A_tilde->A->nrow, igo_cm->A->A->ncol, A_tilde->A->nzmax,
                            A_tilde, igo_cm);
    if(res != 1) { return 0; }

    int* A_tilde_p = (int*) A_tilde->A->p;
    int* A_tilde_i = (int*) A_tilde->A->i;
    double* A_tilde_x = (double*) A_tilde->A->x;
    int* Ap = (int*) igo_cm->A->A->p;
    int* Ai = (int*) igo_cm->A->A->i;
    double* Ax = (double*) igo_cm->A->A->x;
    for(int j = 0; j < A_tilde->A->ncol; j++) {
        int A_tilde_col_start = A_tilde_p[j];
        int A_tilde_col_end = A_tilde_p[j + 1];
        int A_col_start = Ap[j];
        int A_col_end = Ap[j + 1];
        for(int A_tilde_idx = A_tilde_col_start; A_tilde_idx < A_tilde_col_end; A_tilde_idx++) {
            int A_idx = A_col_start;
            while(A_tilde_i[A_tilde_idx] != Ai[A_idx]) {
                A_idx++;
                if(A_idx == A_col_end) { return 2; }    // If cannot find matching row in A, return error code
            }

            double tmp_val = Ax[A_idx];
            Ax[A_idx] = A_tilde_x[A_tilde_idx];
            A_tilde_x[A_tilde_idx] = tmp_val;
            
        }
    }
    
    // 7. igo_updown_solve(-1, new_A, L, y, delta_Ab)
    res = igo_updown_solve(-1, A_tilde, igo_cm->L, igo_cm->y, delta_Ab, igo_cm);
    assert(res == 1);

    // 8. triangular solve DL.T x = y
    igo_dense* x_new = igo_solve(CHOLMOD_DLt, igo_cm->L, igo_cm->y, igo_cm);
    igo_free_dense(&igo_cm->x, igo_cm);
    igo_cm->x = x_new;


    // 9. Release all memory allocated
    igo_free_sparse(&new_Ab, igo_cm);

    return 1;
}

void igo_print_cholmod_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_sparse* A,
    cholmod_common* cholmod_cm
) {
    cholmod_print_sparse(A, name, cholmod_cm);

    if(!verbose) 
        return; 

    if(verbose >= 2) {

        printf("nrow = %d, ncol = %d, nzmax = %d\n", A->nrow, A->ncol, A->nzmax);

        // Access the data arrays
        int* Ap = (int*) A->p;
        int* Ai = (int*) A->i;
        double* Ax = (double*) A->x;
        printf("Ap = ");       
        for(int j = 0; j < A->ncol + 1; j++) {
            printf("%d ", Ap[j]);
        }
        printf("\n");

        printf("Ai = ");       
        for(int i = 0; i < Ap[A->ncol]; i++) {
            printf("%d ", Ai[i]);
        }
        printf("\n");
        printf("Ax = ");       
        for(int i = 0; i < Ap[A->ncol]; i++) {
            printf("%f ", Ax[i]);
        }
        printf("\n");
    }

    // Access the data arrays
    int* Ap = (int*) A->p;
    int* Ai = (int*) A->i;
    double* Ax = (double*) A->x;

    // Iterate through the columns
    for (int j = 0; j < A->ncol; j++) {
        int start = Ap[j];
        int end = Ap[j + 1];

        // Iterate through the non-zero entries in the current column
        for (int i = start; i < end; i++) {

            double value = Ax[i];
            int row = Ai[i];
            // printf("row = %d \n", row);

            printf("Value at (%d, %d) = %f\n", row, j, value);
        }
    }
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

    // Only print LL' factorization
    cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, cholmod_cm);

    if(verbose >= 3) {
        printf("itype = %d, xtype = %d, dtype = %d\n", L->itype, L->xtype, L->dtype);
        printf("ordering = %d, is_ll = %d, is_super = %d, is_monotonic = %d\n", L->ordering, L->is_ll, L->is_super, L->is_monotonic);
    }

    if(verbose >= 2) {
        int* Lp = (int*) L->p;
        int* Li = (int*) L->i;
        double* Lx = (double*) L->x;
        int* LPerm = (int*) L->Perm;
        int* Lnext = (int*) L->next;
        int* Lprev = (int*) L->prev;

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

            // Iterate through the columns
            for (int j = 0; j < L->n; j++) {
                int start = column_pointers[j];
                int end = column_pointers[j + 1];

                // Iterate through the non-zero entries in the current column
                for (int i = start; i < end; i++) {

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

    cholmod_change_factor(CHOLMOD_REAL, is_ll_old, 0, 1, 1, L, cholmod_cm);
}

