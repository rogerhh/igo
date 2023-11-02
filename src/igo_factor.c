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
    int nzmax_alloc,
    cholmod_factor* L
) {
    L->i = realloc(L->i, nzmax_alloc * sizeof(int));
    L->x = realloc(L->x, nzmax_alloc * sizeof(double));
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
    igo_L->n_alloc = 32;
    igo_L->nzmax_alloc = 64;
    igo_L->L = cholmod_allocate_factor(0, igo_cm->cholmod_cm);

    cholmod_factor* L = igo_L->L;

    cholmod_change_factor(CHOLMOD_REAL, 0, 0, 1, 1, L, igo_cm->cholmod_cm);

    factor_alloc_n(igo_L->n_alloc, igo_L->L);
    factor_alloc_nzmax(igo_L->nzmax_alloc, igo_L->L);

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
    igo_L->nzmax_alloc = L->nzmax;
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

    if(igo_L->nzmax_alloc < nzmax) {
        do {
            igo_L->nzmax_alloc *= 2;
        }
        while(igo_L->nzmax_alloc < nzmax);

        factor_alloc_nzmax(igo_L->nzmax_alloc, L);
    }

    /* L->nzmax should track exactly how much memory is allocated */
    L->nzmax = igo_L->nzmax_alloc;

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
            Lp[j + 1] = Lp[j] + 1;
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

// int igo_factor_adjust_nzmax(igo_factor* igo_L, igo_common* igo_cm) {
//   cholmod_factor* L = igo_L->L;
//   int new_nzmax = L->nzmax;
//   if(new_nzmax >= igo_L->nzmax_alloc) {
//     igo_resize_factor(L->n, new_nzmax, )
//   }
//   return 1;
// }

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
        igo_print_factor(3, "igo_L in updown", igo_L, igo_cm);
    }
    cholmod_updown(update, A, L, igo_cm->cholmod_cm);
    /* L may have been reallocated, therefore we must update nzmax_alloc */
    igo_L->nzmax_alloc = L->nzmax;
    igo_print_factor(3, "igo_L in updown after", igo_L, igo_cm);
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
    printf("In igo_factor.c A_nrow = %d, A_ncol = %d, B_nrow = %d\n", delta_A->A->nrow, delta_A->A->ncol, igo_delta_b->B->nrow);
    fflush(stdout);
    assert(delta_A->A->nrow == igo_delta_b->B->nrow);
    assert(igo_x->B->ncol <= 1);
    assert(igo_delta_b->B->ncol == 1);

    cholmod_factor* L = igo_L->L;
    cholmod_dense* x = igo_x->B;
    cholmod_dense* delta_b = igo_delta_b->B;
    int delta_A_nrow = delta_A->A->nrow;
    if(delta_A_nrow > L->n) {
        int newrow = delta_A_nrow - L->n;
        printf("Before resize factor\n");
        fflush(stdout);
        igo_resize_factor(delta_A_nrow, L->nzmax + newrow, igo_L, igo_cm);
    }
    if(delta_A_nrow > x->nrow) {
        printf("Before resize dense %d\n", delta_A_nrow);
        fflush(stdout);
        igo_resize_dense(delta_A_nrow, 1, delta_A_nrow, igo_x, igo_cm);
        printf("After resize dense %d\n", delta_A_nrow);
        fflush(stdout);
    }
    // igo_print_factor(3, "L before updown solve", igo_L, igo_cm);
    // fflush(stdout);
    printf("L size %d, b size %d\n", igo_L->L->n, delta_b->nrow);
    fflush(stdout);
    igo_print_factor(0, "L before updown solve", igo_L, igo_cm);
    printf("Before updown solve\n");
    fflush(stdout);
    // return cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
    cholmod_updown_solve(update, delta_A->A, L, x, delta_b, igo_cm->cholmod_cm);
    /* L may have been reallocated, therefore we must update nzmax_alloc */
    igo_L->nzmax_alloc = L->nzmax;
    printf("After updown solve\n");
    printf("L size %d, b size %d\n", igo_L->L->n, delta_b->nrow);
    igo_print_factor(3, "L after updown solve", igo_L, igo_cm);
    fflush(stdout);
    igo_print_factor(0, "L after updown solve", igo_L, igo_cm);
    fflush(stdout);
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
