#include "igo.h"

#include <assert.h>

/* Initialize an igo_sparse_matrix */
igo_sparse* igo_allocate_sparse (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_sparse* igo_A = malloc(sizeof(igo_sparse));
    igo_A->ncol_alloc = 32;
    igo_A->nzmax_alloc = 64;
    igo_A->A = 
        cholmod_allocate_sparse(0, 0, 0, true, true, 0, CHOLMOD_REAL, igo_cm->cholmod_cm);

    cholmod_sparse* A = igo_A->A;

    A->p = realloc(A->p, igo_A->ncol_alloc * sizeof(int));
    A->i = realloc(A->i, igo_A->nzmax_alloc * sizeof(int));
    A->x = realloc(A->x, igo_A->nzmax_alloc * sizeof(double));

    igo_resize_sparse(nrow, ncol, nzmax, igo_A, igo_cm);
    return igo_A;
}

void igo_free_sparse (
    /* --- in/out --- */
    igo_sparse** igo_A_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    if(igo_A_handle == NULL) 
        return; 

    igo_sparse* igo_A = *igo_A_handle;
    if(igo_A == NULL) {
        return;
    }

    cholmod_sparse* A = igo_A->A;

    free(A->p);
    free(A->i);
    free(A->x);
    free(A->nz);
    free(A->z);

    A->p = NULL;
    A->i = NULL;
    A->x = NULL;
    A->nz = NULL;
    A->z = NULL;

    cholmod_free_sparse(&(igo_A->A), igo_cm->cholmod_cm);
    free(*igo_A_handle);
    *igo_A_handle = NULL;
}

/* Resize an igo_sparse A to (nrow, ncol, nzmax)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes */
int igo_resize_sparse (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_sparse* A = igo_A->A;

    if(igo_A->ncol_alloc < ncol) {
        do {
            igo_A->ncol_alloc *= 2;
        }
        while(igo_A->ncol_alloc < ncol);
        A->p = realloc(A->p, (igo_A->ncol_alloc + 1) * sizeof(int));
    }
    assert(igo_A->ncol_alloc >= ncol);

    if(igo_A->nzmax_alloc < nzmax) {
        do {
            igo_A->nzmax_alloc *= 2;
        }
        while(igo_A->nzmax_alloc < ncol);
        A->i = realloc(A->i, igo_A->nzmax_alloc * sizeof(int));
        A->x = realloc(A->x, igo_A->nzmax_alloc * sizeof(double));
    }

    assert(igo_A->nzmax_alloc >= nzmax);

    int old_ncol = A->ncol;
    A->nrow = nrow;
    A->ncol = ncol;
    A->nzmax = nzmax;

    int* Ap = (int*) A->p;
    for(int i = old_ncol + 1; i <= ncol; i++) {
        Ap[i] = Ap[old_ncol];
    }

}

/* Perform igo_A->A = [igo_A->A B]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_horzappend_sparse (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) {
    // TODO: Error checking. We are assuming both matrices to be packed 
    cholmod_sparse* A = igo_A->A;
    assert(B->nrow >= A->nrow);

    int newrow = B->nrow;
    int oldcol = A->ncol;
    int newcol = A->ncol + B->ncol;
    int oldnzmax = A->nzmax;
    int newnzmax = A->nzmax + B->nzmax;

    igo_resize_sparse(newrow, newcol, newnzmax, igo_A, igo_cm);

    int* Ap = (int*) A->p;
    int* Bp = (int*) B->p;
    int old_maxAp = Ap[oldcol];
    for(int i = oldcol; i < newcol; i++) {
        Ap[i + 1] = Bp[i - oldcol + 1] + old_maxAp;
    }

    int copy_size = newnzmax - oldnzmax;
    memcpy(A->i + old_maxAp * sizeof(int), B->i, copy_size * sizeof(int));
    memcpy(A->x + old_maxAp * sizeof(double), B->x, copy_size * sizeof(double));
    
}

void igo_print_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    igo_sparse* igo_A,
    igo_common* igo_cm
) {
    igo_print_cholmod_sparse(verbose, name, igo_A->A, igo_cm->cholmod_cm);
}
