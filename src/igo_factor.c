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

    igo_factor* igo_L = malloc(sizeof(igo_factor));
    igo_L->n_alloc = igo_cm->FACTOR_NCOL_ALLOC;
    igo_L->L = cholmod_allocate_factor(0, igo_cm->cholmod_cm);

    cholmod_factor* L = igo_L->L;

    cholmod_change_factor(CHOLMOD_REAL, 0, 0, 1, 1, L, igo_cm->cholmod_cm);

    factor_alloc_n(igo_L->n_alloc, igo_L->L);
    factor_alloc_nzmax(igo_cm->FACTOR_NZMAX_ALLOC, igo_L->L, igo_cm->cholmod_cm);

    igo_resize_factor(n, nzmax, igo_L, igo_cm);

    return igo_L;
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
    igo_factor* igo_L = malloc(sizeof(igo_factor));
    igo_L->n_alloc = L->n;
    igo_L->L = L;

    cholmod_change_factor(CHOLMOD_REAL, 0, 0, 1, 1, L, igo_cm->cholmod_cm);

    *L_handle = NULL;

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
            Lx[Lp[j]] = 1e-12;
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

// int igo_updown2 (
//     /* --- input --- */
//     igo_sparse* igo_C,
//     igo_sparse* igo_D,
//     /* --- in/out --- */
//     igo_factor* igo_L,
//     /* ------------- */
//     igo_common* igo_cm
// ) {
//     cholmod_factor* L = igo_L->L;
//     cholmod_sparse* C = igo_C->A;
//     cholmod_sparse* D = igo_C->A;
//     int new_max_row = max(max(C->nrow, D->nrow), L->n);
//     if(new_max_row > L->n) {
//         igo_resize_factor(new_max_row, L->nzmax, igo_L, igo_cm);
//     }
//     cholmod_updown2(C, D, L, igo_cm->cholmod_cm);
//     return 1;
// }

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
    cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
    return 1;
}

// int igo_updown_solve2 (
//     /* --- input --- */
//     igo_sparse* igo_C,
//     igo_sparse* igo_D,
//     /* --- in/out --- */
//     igo_factor* igo_L,
//     igo_dense* igo_x,
//     igo_dense* igo_delta_b,
//     /* ------------- */
//     igo_common* igo_cm
// ) {
//     assert(igo_C->A->nrow == igo_delta_b->B->nrow);
//     assert(igo_D->A->nrow == igo_delta_b->B->nrow);
//     assert(igo_x->B->ncol <= 1);
//     assert(igo_delta_b->B->ncol == 1);
// 
//     cholmod_factor* L = igo_L->L;
//     cholmod_dense* x = igo_x->B;
//     cholmod_dense* delta_b = igo_delta_b->B;
//     int delta_A_nrow = delta_A->A->nrow;
//     if(delta_A_nrow > L->n) {
//         int newrow = delta_A_nrow - L->n;
//         igo_resize_factor(delta_A_nrow, L->nzmax + newrow, igo_L, igo_cm);
//     }
//     if(delta_A_nrow > x->nrow) {
//         igo_resize_dense(delta_A_nrow, 1, delta_A_nrow, igo_x, igo_cm);
//     }
//     // return cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
//     cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
//     return 1;
// }

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
    cholmod_dense* x = cholmod_solve(sys, igo_L->L, igo_B->B, igo_cm->cholmod_cm);
    return igo_allocate_dense2(&x, igo_cm);
}

void igo_print_factor (
    /* --- input --- */
    int verbose,
    char* name,
    igo_factor* igo_L,
    igo_common* igo_cm
) {
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
    cholmod_common* igo_cm
) {
    if(L1 == NULL && L2 == NULL) { return true; }
    if(L1 == NULL || L2 == NULL) { return false; }
    if(L1->n != L2->n) { return false; }
    int* L1p = L1->p;
    int* L1i = L1->i;
    int* L1nz = L1->nz;
    int* L2p = L2->p;
    int* L2i = L2->i;
    int* L2nz = L2->nz;
    double* L1x = (double*) L1->x;
    double* L2x = (double*) L2->x;
    for(int j = 0; j < L1->n; j++) {
        if(L1nz[j] != L2nz[j]) { return false; }
        int idx1 = L1p[j];
        int idx2 = L2p[j];
        for(int cnt = 0; cnt < L1nz[j]; cnt++, idx1++, idx2++) {
            if(L1i[idx1] != L2i[idx2]) { return false; }
            if(fabs(L1x[idx1] - L2x[idx2]) >= eps) { return false; }
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
