#include "igo.h"

#include <assert.h>
#include <stdio.h>

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
    igo_A->ncol_alloc = IGO_SPARSE_DEFAULT_NCOL_ALLOC;
    igo_A->nzmax_alloc = IGO_SPARSE_DEFAULT_NZMAX_ALLOC;
    igo_A->A = 
        cholmod_allocate_sparse(0, 0, 0, true, true, 0, CHOLMOD_REAL, igo_cm->cholmod_cm);

    cholmod_sparse* A = igo_A->A;

    A->p = realloc(A->p, igo_A->ncol_alloc * sizeof(int));
    A->i = realloc(A->i, igo_A->nzmax_alloc * sizeof(int));
    A->x = realloc(A->x, igo_A->nzmax_alloc * sizeof(double));

    igo_resize_sparse(nrow, ncol, nzmax, igo_A, igo_cm);
    return igo_A;
}

/* Initialize an igo_sparse_matrix */
igo_sparse* igo_allocate_sparse2 (
    /* --- inout --- */
    cholmod_sparse** A_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_sparse* igo_A = malloc(sizeof(igo_sparse));
    cholmod_sparse* A = *A_handle;
    igo_A->ncol_alloc = A->ncol;
    igo_A->nzmax_alloc = A->nzmax;
    igo_A->A = A;

    *A_handle = NULL;

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
 * to accomodate for future resizes 
 * REQUIRES: nrow >= igo_A->A->nrow
 * */
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
    
    // Error checking
    if(nrow < A->nrow) {
        return 0;
    }

    if(igo_A->ncol_alloc < ncol) {
        // igo_A->ncol_alloc may not be initialized correctly if it is allocated from 
        // igo_allocate_sparse2. So we initialize it here to default / 2, and it will 
        // double from there
        igo_A->ncol_alloc = igo_A->ncol_alloc == 0? 
          IGO_SPARSE_DEFAULT_NCOL_ALLOC / 2 : igo_A->ncol_alloc;
        do {
            igo_A->ncol_alloc *= 2;
        }
        while(igo_A->ncol_alloc < ncol);
        A->p = realloc(A->p, (igo_A->ncol_alloc + 1) * sizeof(int));
    }
    assert(igo_A->ncol_alloc >= ncol);

    if(igo_A->nzmax_alloc < nzmax) {
      igo_A->nzmax_alloc = igo_A->nzmax_alloc == 0?
          IGO_SPARSE_DEFAULT_NZMAX_ALLOC / 2 : igo_A->nzmax_alloc;
        do {
            igo_A->nzmax_alloc *= 2;
        }
        while(igo_A->nzmax_alloc < nzmax);
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

    return 1;

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

    printf("igo_horzappend_sparse before resize\n");
    fflush(stdout);

    igo_resize_sparse(newrow, newcol, newnzmax, igo_A, igo_cm);

    printf("igo_horzappend_sparse after resize\n");
    fflush(stdout);

    int* Ap = (int*) A->p;
    int* Bp = (int*) B->p;
    int old_maxAp = Ap[oldcol];
    for(int i = oldcol; i < newcol; i++) {
        Ap[i + 1] = Bp[i - oldcol + 1] + old_maxAp;
    }

    int copy_size = newnzmax - oldnzmax;
    memcpy(A->i + old_maxAp * sizeof(int), B->i, copy_size * sizeof(int));
    memcpy(A->x + old_maxAp * sizeof(double), B->x, copy_size * sizeof(double));

    printf("In horzappend, after resize nrow = %d ncol = %d\n", A->nrow, A->ncol);
    
    return 1;
}

/* Perform igo_A->A = [igo_A->A B]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_horzappend_sparse2 (
    /* --- input --- */
    igo_sparse* igo_B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_horzappend_sparse(igo_B->A, igo_A, igo_cm);
}

/* Performs igo_A = [igo_A; B]. 
 * This is needed because cholmod_vertcat makes copies of the inputs */
int igo_vertappend_sparse (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) {
    // TODO: Error checking. Need to make sure both matrices are packed
    cholmod_sparse* A = igo_A->A;

    int newrow = A->nrow + B->nrow;
    int oldrow = A->nrow;
    int newcol = A->ncol > B->ncol? A->ncol : B->ncol;
    int oldnzmax = A->nzmax;
    int newnzmax = A->nzmax + B->nzmax;

    int* Ap = (int*) A->p;
    int* Bp = (int*) B->p;

    igo_resize_sparse(newrow, newcol, newnzmax, igo_A, igo_cm);

    // First sort out the column ptrs
    int prev_col_start = 0;
    for(int j = 0; j < B->ncol; j++) {
        int old_Aj_size = Ap[j + 1] - prev_col_start;
        int Bj_size = Bp[j + 1] - Bp[j];
        prev_col_start = Ap[j + 1];
        Ap[j + 1] = Ap[j] + old_Aj_size + Bj_size;
    }

    // Then copy over data from last column to first
    int* Ai = (int*) A->i;
    int* Bi = (int*) B->i;
    double* Ax = (double*) A->x;
    double* Bx = (double*) B->x;
    for(int j = B->ncol - 1; j >= 0; j--) {
        int Apj = Ap[j];
        int new_col_size = Ap[j + 1] - Apj;

        int Bpj = Bp[j], Bpj_1 = Bp[j + 1];
        int Bj_size = Bpj_1 - Bpj;
        int old_Aj_size = new_col_size - Bj_size;

        // Copy over B data
        memcpy(Ax + Apj + old_Aj_size, Bx + Bpj, Bj_size * sizeof(double));

        // Copy over B indices but shift by number of old rows
        for(int i = 0; i < Bj_size; i++) {
          Ai[Apj + old_Aj_size + i] = Bi[Bpj + i] + oldrow;
        }

        // Copy over A data
        prev_col_start -= old_Aj_size;
        memmove(Ax + Apj, Ax + prev_col_start, old_Aj_size * sizeof(double));

        // Copy over A indices as is
        memmove(Ai + Apj, Ai + prev_col_start, old_Aj_size * sizeof(int));
    }

    return 1;
}

/* Performs igo_A = [igo_A igo_B]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_vertappend_sparse2 (
    /* --- input --- */
    igo_sparse* igo_B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) {
  return igo_vertappend_sparse(igo_B->A, igo_A, igo_cm);
}

igo_sparse* igo_ssmult (
    /* --- input --- */
    igo_sparse* igo_A,
    igo_sparse* igo_B,
    igo_common* igo_cm
) {
    cholmod_sparse* C = cholmod_ssmult(igo_A->A, igo_B->A, 0, true, true, igo_cm->cholmod_cm);
    return igo_allocate_sparse2(&C, igo_cm);
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
