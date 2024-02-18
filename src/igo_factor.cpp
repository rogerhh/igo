#include "cholmod.h"
#include "igo.h"

#include <assert.h>

// Helper function to allocate everything that depends on n
static int factor_alloc_n (
    int n_alloc,
    cholmod_factor* L
) {
    L->Perm = realloc(L->Perm, n_alloc * sizeof(int));
    L->ColCount = realloc(L->ColCount, n_alloc * sizeof(int));
    L->IPerm = realloc(L->IPerm, n_alloc * sizeof(int));
    L->p = realloc(L->p, (n_alloc + 1) * sizeof(int));
    L->nz = realloc(L->nz, n_alloc * sizeof(int));
    L->next = realloc(L->next, (n_alloc + 2) * sizeof(int));
    L->prev = realloc(L->prev, (n_alloc + 2) * sizeof(int));

    return 1;
}

// Helper function to allocate everything that depends on nzmax
static int factor_alloc_nzmax (
    int nzmax,
    cholmod_factor* L,
    cholmod_common* cholmod_cm
) {
    cholmod_reallocate_factor(nzmax, L, cholmod_cm);
    return 1;
}

/* Initialize an igo_factor */
igo_factor* igo_allocate_factor (
    /* --- input --- */
    int n,
    int nzmax,
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_allocate_identity_factor(n, nzmax, 1e-12, igo_cm);
}

/* Initialize an igo_factor from an existing cholmod_factor
 * Destroys the original factor pointer */
igo_factor* igo_allocate_factor2 (
    /* --- input --- */
    cholmod_factor** L_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_factor* L = *L_handle;
    printf("in allocate factor2\n");
    fflush(stdout);
    igo_factor* igo_L = (igo_factor*) malloc(sizeof(igo_factor));
    printf("done allocate factor2\n");
    fflush(stdout);
    igo_L->n_alloc = L->n;
    igo_L->L = L;

    // cholmod_change_factor(CHOLMOD_REAL, 0, 0, 1, 1, L, igo_cm->cholmod_cm);

    *L_handle = NULL;

    return igo_L;
}

/* Initialize an igo_factor that only has 1s on the diagonal */
igo_factor* igo_allocate_identity_factor (
    /* --- input --- */
    int n,
    int nzmax,
    double d,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_factor* igo_L = (igo_factor*) malloc(sizeof(igo_factor));
    igo_L->n_alloc = igo_cm->FACTOR_NCOL_ALLOC;
    igo_L->L = cholmod_allocate_factor(0, igo_cm->cholmod_cm);

    cholmod_factor* L = igo_L->L;

    cholmod_change_factor(CHOLMOD_REAL, 0, 0, 1, 1, L, igo_cm->cholmod_cm);

    factor_alloc_n(igo_L->n_alloc, igo_L->L);
    factor_alloc_nzmax(igo_cm->FACTOR_NZMAX_ALLOC, igo_L->L, igo_cm->cholmod_cm);

    igo_resize_factor2(n, nzmax, d, igo_L, igo_cm);

    return igo_L;
}

int igo_resize_factor (
    /* --- input --- */
    int n,
    int nzmax,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_resize_factor2(n, nzmax, 1e-12, igo_L, igo_cm);
}

int igo_resize_factor2 (
    /* --- input --- */
    int n,
    int nzmax,
    double d,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_factor* L = igo_L->L;

    // cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, igo_cm->cholmod_cm);

    // igo_print_cholmod_factor(3, "L after change", L, igo_cm->cholmod_cm);

    int nold = L->n;
    L->n = n;
    L->minor = n;

    if(igo_L->n_alloc < n) {
        do {
            igo_L->n_alloc *= 2;
        }
        while(igo_L->n_alloc < n);

        factor_alloc_n(igo_L->n_alloc, L);
    }

    int new_nnz = nzmax > L->nzmax? nzmax - L->nzmax : 0;
    int new_cols = L->n > nold? L->n - nold : 0;
    int new_cols_nnz = new_cols * igo_cm->FACTOR_DEFAULT_COL_SIZE;
    int new_nnz_needed = new_nnz + new_cols_nnz;
    int* Lp = (int*) L->p;
    int nzmax_needed = Lp[nold] + new_nnz_needed;
    if(nzmax_needed > L->nzmax) {
        int new_nzmax = L->nzmax;
        do {
            new_nzmax *= 2;
        }
        while(new_nzmax < nzmax_needed);

        factor_alloc_nzmax(new_nzmax, L, igo_cm->cholmod_cm);
    }

    int* LPerm = (int*) L->Perm;
    int* LColCount = (int*) L->ColCount;

    // New rows and cols use the natural permutation
    for(int j = nold; j < n; j++) {
        LPerm[j] = j;
        LColCount[j] = 1;
    }

    if(n != nold) {
        // Only need to set the new columns if there are new columns
        int* Lp = (int*) L->p;
        int* Li = (int*) L->i;
        double* Lx = (double*) L->x;
        int* Lnz = (int*) L->nz;

        int* Lnext = (int*) L->next;
        int* Lprev = (int*) L->prev;

        int oldfirstcol;
        int oldlastcol;
        if(nold == 0) {
          oldfirstcol = 0;
          oldlastcol = 0;
        }
        else{
            oldfirstcol = Lnext[nold + 1];
            oldlastcol = Lprev[nold];
        }

        // Add 1e-12 * I to extend the diagonal
        for(int j = nold; j < n; j++) {
            Lp[j + 1] = Lp[j] + igo_cm->FACTOR_DEFAULT_COL_SIZE;
            Li[Lp[j]] = j;
            Lx[Lp[j]] = d;
            Lnz[j] = 1;
        }

        for(int j = nold; j < n; j++) {
            Lnext[j] = j + 1;
            Lprev[j + 1] = j;
        }

        Lnext[n + 1] = oldfirstcol;  /* Next col of the head should be the first col */
        Lprev[n + 1] = -1;           /* Prev col of the head should be -1 */

        Lnext[n] = -1;               /* Next col of the tail should be -1 */

        Lprev[oldfirstcol] = n + 1;  /* Prev col of the first col should be the head */
        // Lnext[oldlastcol] = nold;    /* Next col of the old last col should be the first added col */
    }

    return 1;
}

void igo_free_factor (
    /* --- in/out --- */
    igo_factor** igo_L_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    if(igo_L_handle == NULL) {
        return;
    }

    igo_factor* igo_L = *igo_L_handle;
    if(igo_L == NULL) {
        return;
    }

    cholmod_factor* L = igo_L->L;

    free(L->Perm);
    free(L->ColCount);
    free(L->IPerm);
    free(L->p);
    free(L->nz);
    free(L->next);
    free(L->prev);
    free(L->i);
    free(L->x);

    L->Perm = NULL;
    L->ColCount = NULL;
    L->IPerm = NULL;
    L->p = NULL;
    L->nz = NULL;
    L->next = NULL;
    L->prev = NULL;
    L->i = NULL;
    L->x = NULL;

    cholmod_free_factor(&(igo_L->L), igo_cm->cholmod_cm);
    free(*igo_L_handle);
    *igo_L_handle = NULL;
    return;
}

/* Copys an igo_factor
 * */
igo_factor* igo_copy_factor (
    /* --- input --- */
    igo_factor* L,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_factor* cholmod_L = cholmod_copy_factor(L->L, igo_cm->cholmod_cm);
    return igo_allocate_factor2(&cholmod_L, igo_cm);
}

/* Combine cholmod_analyze and cholmod_factorize in one step
 * Takes in a sparse matrix PA 
 * */
igo_factor* igo_analyze_and_factorize (
    /* --- input --- */
    igo_sparse* A,
    /* ------------- */
    igo_common* igo_cm
) {

        printf("Before analyze\n");
        fflush(stdout);
    cholmod_factor* cholmod_L = cholmod_analyze(A->A, igo_cm->cholmod_cm);
        printf("Before factorize\n");
        fflush(stdout);
    printf("nzmax = %d\n", cholmod_L->nzmax);
        fflush(stdout);
    cholmod_factorize2(A->A, cholmod_L, igo_cm->cholmod_cm);
        printf("Done factorize\n");
        fflush(stdout);
    printf("nzmax = %d\n", cholmod_L->nzmax);
        fflush(stdout);
    
    igo_factor* igo_L = igo_allocate_factor2(&cholmod_L, igo_cm);
    printf("igo_L = %p nzmax = %d\n", igo_L, igo_L->L->nzmax);
        fflush(stdout);

    igo_print_factor(2, "symbolic L", igo_L, igo_cm);
        fflush(stdout);
    return igo_L;
}

int igo_updown (
    /* --- input --- */
    int update,             // 1 for update, 0 for downdate
    igo_sparse* igo_A,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_factor* L = igo_L->L;
    cholmod_sparse* A = igo_A->A;
    int A_nrow = A->nrow;
    if(A_nrow > L->n) {
        int newrow = A_nrow - L->n;
        igo_resize_factor(A_nrow, L->nzmax + newrow, igo_L, igo_cm);
    }
    cholmod_updown(update, A, L, igo_cm->cholmod_cm);
    return 1;
}

int igo_updown2 (
    /* --- input --- */
    igo_sparse* igo_C,
    igo_sparse* igo_D,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_factor* L = igo_L->L;
    cholmod_sparse* C = igo_C->A;
    cholmod_sparse* D = igo_C->A;
    cholmod_updown2(C, D, L, igo_cm->cholmod_cm);
    return 1;
}

int igo_updown_solve (
    /* --- input --- */
    int update,             // 1 for update, 0 for downdate
    igo_sparse* delta_A,
    /* --- in/out --- */
    igo_factor* igo_L,
    igo_dense* igo_x,
    igo_dense* igo_delta_b,
    /* ------------- */
    igo_common* igo_cm
) {
    assert(delta_A->A->nrow == igo_delta_b->B->nrow);
    assert(igo_x->B->ncol <= 1);
    assert(igo_delta_b->B->ncol == 1);

    cholmod_factor* L = igo_L->L;
    cholmod_dense* x = igo_x->B;
    cholmod_dense* delta_b = igo_delta_b->B;
    int delta_A_nrow = delta_A->A->nrow;
    if(delta_A_nrow > L->n) {
        int newrow = delta_A_nrow - L->n;
        igo_resize_factor(delta_A_nrow, L->nzmax + newrow, igo_L, igo_cm);
    }
    if(delta_A_nrow > x->nrow) {
        igo_resize_dense(delta_A_nrow, 1, delta_A_nrow, igo_x, igo_cm);
    }
    // return cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
    cholmod_updown3_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
    return 1;
}

int igo_updown2_solve (
    /* --- input --- */
    igo_sparse* igo_C,
    igo_sparse* igo_D,
    /* --- in/out --- */
    igo_factor* igo_L,
    igo_dense* igo_x,
    igo_dense* igo_delta_b,
    /* ------------- */
    igo_common* igo_cm
) {
    assert(igo_C->A->nrow == igo_delta_b->B->nrow);
    assert(igo_D->A->nrow == igo_delta_b->B->nrow);
    assert(igo_x->B->ncol <= 1);
    assert(igo_delta_b->B->ncol == 1);

    cholmod_factor* L = igo_L->L;
    cholmod_dense* x = igo_x->B;
    cholmod_dense* delta_b = igo_delta_b->B;

    cholmod_updown2_solve(igo_C->A, igo_D->A, L, x, delta_b, igo_cm->cholmod_cm);
    return 1;
}

/* Wrapper around cholmod_solve
 * */
igo_dense* igo_solve (
    /* --- input --- */
    int sys,            // System to solve
    igo_factor* igo_L,  // Cholesky factorization
    igo_dense* igo_B,   // Right hand side matrix
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* X = cholmod_solve(sys, igo_L->L, igo_B->B, igo_cm->cholmod_cm);
    return igo_allocate_dense2(&X, igo_cm);
}

void igo_print_cholmod_factor(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_factor* L,
    cholmod_common* cholmod_cm
) {
    cholmod_factor* L_copy = cholmod_copy_factor(L, cholmod_cm);
    cholmod_print_factor(L_copy, name, cholmod_cm);

    bool is_ll_old = L->is_ll;

    // // Only print LL' factorization
    // cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, cholmod_cm);

    if(verbose >= 1) {
        printf("itype = %d, xtype = %d, dtype = %d\n", L->itype, L->xtype, L->dtype);
        printf("ordering = %d, is_ll = %d, is_super = %d, is_monotonic = %d\n", L->ordering, L->is_ll, L->is_super, L->is_monotonic);
        printf("nzmax = %d\n", L->nzmax);
        fflush(stdout);
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
        printf("Lp+1 - Lp = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", Lp[Lnext[j]] - Lp[j]);
        }
        printf("\n");
        // printf("Li = ");
        // for(int j = 0; j < Lp[L->n]; j++) {
        //     printf("%d ", Li[j]);
        // }
        // printf("\n");
        // printf("Lx = ");
        // for(int j = 0; j < Lp[L->n]; j++) {
        //     printf("%f ", Lx[j]);
        // }
        // printf("\n");
        printf("nz = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", Lnz[j]);
        }
        printf("\n");
        printf("Slack = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", Lp[Lnext[j]] - Lp[j] - Lnz[j]);
        }
        printf("\n");
        // printf("Next = ");
        // for(int j = 0; j < L->n + 2; j++) {
        //     printf("%d ", Lnext[j]);
        // }
        // printf("\n");
        // printf("Prev = ");
        // for(int j = 0; j < L->n + 2; j++) {
        //     printf("%d ", Lprev[j]);
        // }
        // printf("\n");
        // printf("Perm = ");
        // for(int j = 0; j < L->n; j++) {
        //     printf("%d ", LPerm[j]);
        // }
        // printf("\n");
        // printf("ColCount = ");
        // for(int j = 0; j < L->n; j++) {
        //     printf("%d ", LColCount[j]);
        // }
        // printf("\n");
    }

    if(verbose >= 3) {
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

void igo_print_factor (
    /* --- input --- */
    int verbose,
    char* name,
    igo_factor* igo_L,
    igo_common* igo_cm
) {
    if(!igo_L) {
        printf("Factor %s is NULL\n", name);
        return;
    }
    printf("Factor %s: n_alloc: %d\n", name, igo_L->n_alloc);
    fflush(stdout);
    igo_print_cholmod_factor(verbose, name, igo_L->L, igo_cm->cholmod_cm);
}

/* Test if two cholmod_factors L1 and L2 are equal, 
 * i.e. nonzero(L1) == nonzero(L2) and |L1 - L2|_infty < eps
 * */
bool igo_cholmod_factor_eq(
    /* --- input --- */
    cholmod_factor* L1,
    cholmod_factor* L2,
    double eps,
    /* ------------- */
    cholmod_common* cholmod_cm
) {
    if(L1 == NULL && L2 == NULL) { return true; }
    if(L1 == NULL || L2 == NULL) { printf("1\n"); return false; }
    if(L1->n != L2->n) { return false; }
    int* L1p = (int*) L1->p;
    int* L1i = (int*) L1->i;
    int* L1nz = (int*) L1->nz;
    int* L2p = (int*) L2->p;
    int* L2i = (int*) L2->i;
    int* L2nz = (int*) L2->nz;
    double* L1x = (double*) L1->x;
    double* L2x = (double*) L2->x;
    for(int j = 0; j < L1->n; j++) {
        if(L1nz[j] != L2nz[j]) { return false; }
        int idx1 = L1p[j];
        int idx2 = L2p[j];
        for(int cnt = 0; cnt < L1nz[j]; cnt++, idx1++, idx2++) {
            if(L1i[idx1] != L2i[idx2]) { printf("3\n"); return false; }
            if(fabs(L1x[idx1] - L2x[idx2]) >= eps) { printf("4\n"); return false; }
        }
    }
    return true;
}

/* Wrapper around igo_cholmod_factor_eq
 * */
bool igo_factor_eq(
    /* --- input --- */
    igo_factor* L1,
    igo_factor* L2,
    double eps,
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_cholmod_factor_eq(L1->L, L2->L, eps, igo_cm->cholmod_cm);
}
