#include "cholmod.h"
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

/* Copys an igo_sparse matrix
 * */
igo_sparse* igo_copy_sparse (
    /* --- in/out --- */
    igo_sparse* A,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_sparse* cholmod_A = cholmod_copy_sparse(A->A, igo_cm->cholmod_cm);
    return igo_allocate_sparse2(&cholmod_A, igo_cm);
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
    
    // Input checking
    if(!A->sorted) {
        assert(0);
        return -1;
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

    int old_nrow = A->nrow;
    int old_ncol = A->ncol;
    A->nrow = nrow;
    A->ncol = ncol;
    A->nzmax = nzmax;

    int* Ap = (int*) A->p;
    for(int i = old_ncol + 1; i <= ncol; i++) {
        Ap[i] = Ap[old_ncol];
    }

    if(old_nrow > nrow) {
        // TODO: Make this unpacked. For now just check that there is only 1 column
        assert(ncol <= 1);
        if(ncol == 1) {
            int* Ai = (int*) A->i;
            for(int i = 0; i < Ap[1]; i++) {
                int row = Ai[i];
                if(row >= nrow) {
                    Ap[1] = i;
                }
            }
        }
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
    // Make sure both matrices are packed and sorted
    // Not that it would not work, but it's not tested yet
    if(!B->packed) {
      assert(0);
      return -1;
    }
    else if(!B->sorted) {
      assert(0);
      return -2;
    }
    else if(!igo_A->A->packed) {
      assert(0);
      return -3;
    }
    else if(!igo_A->A->sorted) {
      assert(0);
      return -4;
    }

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
    int stype,
    int values,
    int sorted,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_sparse* C = cholmod_ssmult(igo_A->A, igo_B->A, 
                                       stype, values, sorted, 
                                       igo_cm->cholmod_cm);
    return igo_allocate_sparse2(&C, igo_cm);
}

/* Wrapper around cholmod_sdmult
 * */
void igo_sdmult (
    /* --- input --- */
    igo_sparse* igo_A,
    int transpose,
    double* alpha,
    double* beta,
    igo_dense* igo_X,
    /* --- output --- */
    igo_dense* igo_Y,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_sdmult(igo_A->A, transpose, alpha, beta, igo_X->B, igo_Y->B, igo_cm->cholmod_cm);
}

/* Replace the nonzero columns of igo_A with corresponding columns in A_tilde
 * Return the replaced submatrix in the same pattern as A_tilde
 * TODO: Currently assume columns of A_tilde have the same pattern as columns of A
 * TODO: Do index checking
 * */
igo_sparse* igo_replace_sparse (
    /* --- input --- */
    igo_sparse* A,
    igo_sparse* A_tilde,
    /* ------------- */
    igo_common* igo_cm
) {

    // printf("%d %d %d %d\n", A_tilde->A->nrow, A_tilde->A->ncol, A->A->nrow, A->A->ncol);
    // assert(A_tilde->A->ncol <= A->A->ncol);
    // assert(A_tilde->A->nrow <= A->A->nrow);
    // assert(A_tilde->A->packed);

    igo_sparse* igo_A_tilde_neg = igo_copy_sparse(A_tilde, igo_cm);
    cholmod_sparse* A_tilde_neg = igo_A_tilde_neg->A;
    assert(A_tilde->A->packed);
    assert(igo_cm->A->A->packed);

    int* A_tilde_p = (int*) A_tilde->A->p;
    int* A_tilde_i = (int*) A_tilde->A->i;
    double* A_tilde_x = (double*) A_tilde->A->x;
    int* A_tilde_neg_p = (int*) A_tilde_neg->p;
    int* A_tilde_neg_i = (int*) A_tilde_neg->i;
    double* A_tilde_neg_x = (double*) A_tilde_neg->x;
    int* Ap = (int*) A->A->p;
    int* Ai = (int*) A->A->i;
    double* Ax = (double*) A->A->x;
    // Loop through all columns of A_tilde, here we assume A and A_tilde are packed
    for(int j = 0; j < A_tilde->A->ncol; j++) {
        int A_tilde_col_start = A_tilde_p[j];
        int A_tilde_col_end = A_tilde_p[j + 1];
        int col_nz = A_tilde_col_end - A_tilde_col_start;
        double* A_col = Ax + Ap[j];
        double* A_tilde_col = A_tilde_x + A_tilde_col_start;
        double* A_tilde_neg_col = A_tilde_neg_x + A_tilde_col_start;
        for(int idx = 0; idx < col_nz; idx++) {
            A_tilde_neg_col[idx] = A_col[idx];
            A_col[idx] = A_tilde_col[idx];
        }
    }
    return igo_A_tilde_neg;
}

igo_sparse* igo_submatrix (
    /* --- input --- */
    igo_sparse* A,
    int* Rset,
    int Rsize,
    int* Cset,
    int Csize,
    int values,
    int sorted,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_sparse* A_sub = cholmod_submatrix(A->A, Rset, Rsize, Cset, Csize, 
                                              values, sorted, igo_cm->cholmod_cm);
    return igo_allocate_sparse2(&A_sub, igo_cm);
}

// Print a cholmod_sparse matrix
// Verbose:
//  0: Print standard cholmod output
//  1: Print Ap, Ai, Ax pointers
//  2: Print entries in triplet form
void igo_print_cholmod_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_sparse* A,
    cholmod_common* cholmod_cm
) {
    cholmod_print_sparse(A, name, cholmod_cm);

    if(verbose >= 1) {

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
            printf("%.8lf ", Ax[i]);
        }
        printf("\n");
    }

    if(verbose >= 2) {
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

            printf("Value at (%d, %d) = %lf\n", row, j, value);
          }
        }
    }

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
