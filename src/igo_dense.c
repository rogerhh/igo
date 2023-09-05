#include "igo.h"

#include <assert.h>

static void dense_alloc_nzmax (
    size_t nzmax_alloc,
    cholmod_dense* B
) {
    B->x = realloc(B->x, nzmax_alloc * sizeof(double));
}

/* Initialize an igo_dense matrix */
igo_dense* igo_allocate_dense (
    /* --- input --- */
    int nrow,
    int ncol,
    int d,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_dense* igo_B = malloc(sizeof(igo_dense));
    igo_B->nzmax_alloc = 32;

    igo_B->B = cholmod_zeros(0, 1, CHOLMOD_REAL, igo_cm->cholmod_cm);

    dense_alloc_nzmax(igo_B->nzmax_alloc, igo_B->B);

    igo_resize_dense(nrow, ncol, d, igo_B, igo_cm);

    return igo_B;
}

void igo_free_dense (
    /* --- in/out --- */
    igo_dense** igo_B_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    if(igo_B_handle == NULL) {
        return;
    }

    igo_dense* igo_B = *igo_B_handle;
    if(igo_B == NULL) {
        return;
    }

    cholmod_dense* B = igo_B->B;

    free(B->x);
    B->x = NULL;

    cholmod_free_dense(&(igo_B->B), igo_cm->cholmod_cm);

    free(*igo_B_handle);
    *igo_B_handle = NULL;

    return;
}

/* Resize an igo_dense B to (nrow, ncol, d)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes */
int igo_resize_dense (
    /* --- input --- */
    size_t nrow,
    size_t ncol,
    size_t d,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) {

    // Error checking
    assert(d >= nrow);

    cholmod_dense* B = igo_B->B;

    size_t nrow_old = B->nrow;
    size_t ncol_old = B->ncol;
    size_t d_old = B->d;
    size_t nzmax_old = B->nzmax;

    size_t nzmax = d * ncol;

    if(igo_B->nzmax_alloc < nzmax) {
        do {
            igo_B->nzmax_alloc *= 2;
        } while(igo_B->nzmax_alloc < nzmax);

        dense_alloc_nzmax(igo_B->nzmax_alloc, B);
    }

    B->nrow = nrow;
    B->ncol = ncol;
    B->nzmax = nzmax;

    double* Bx = (double*) B->x;

    // If using up reserved space, need to shift columns
    if(d > d_old) {
        B->d = d;
        double* old_col_start = Bx + ncol_old * d_old;
        double* new_col_start = Bx + ncol_old * d;
        for(int j = ncol_old - 1; j >= 0; j--) {
            old_col_start -= d_old;
            new_col_start -= d;
            memmove(new_col_start, old_col_start, nrow_old * sizeof(double));
            memset(new_col_start + nrow_old, 0, (nrow - nrow_old) * sizeof(double));
        }
    }

}

/* Perform [igo_B->B; Bhat] = [igo_B->B; Bhat]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_vertappend_dense (
    /* --- input --- */
    cholmod_dense* Bhat,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* B = igo_B->B;
    size_t nrow_old = B->nrow;
    size_t ncol_old = B->ncol;
    size_t d_old = B->d;
    size_t nrow_new = B->nrow + Bhat->nrow;
    size_t ncol_new = B->ncol > Bhat->ncol? B->ncol : Bhat->ncol;
    size_t d_new = B->d >= nrow_new? B->d : nrow_new + 16;   // TODO: Make this a parameter

    igo_resize_dense(nrow_new, ncol_new, d_new, igo_B, igo_cm);

    double* old_col_start = (double*) Bhat->x;
    double* new_col_start = (double*) B->x + nrow_old;
    for(size_t j = 0; j < Bhat->ncol; j++) {
        memcpy(new_col_start, old_col_start, Bhat->nrow * sizeof(double));
        old_col_start += Bhat->d;
        new_col_start += B->d;
    }

}

void igo_print_dense(
    /* --- input --- */
    int verbose,
    char* name,
    igo_dense* igo_B,
    igo_common* igo_cm
) {
    igo_print_cholmod_dense(verbose, name, igo_B->B, igo_cm->cholmod_cm);
}