#include "igo.h"

#include <assert.h>

static void dense_alloc_nzmax (
    int nzmax_alloc,
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
    igo_dense* igo_B = (igo_dense*) malloc(sizeof(igo_dense));
    igo_B->nzmax_alloc = 32;

    igo_B->B = cholmod_zeros(0, 1, CHOLMOD_REAL, igo_cm->cholmod_cm);

    dense_alloc_nzmax(igo_B->nzmax_alloc, igo_B->B);

    igo_resize_dense(nrow, ncol, d, igo_B, igo_cm);

    return igo_B;
}

/* Initialize an igo_dense matrix with and existing dense matrix
 * Destroys the original dense matrix pointer */
igo_dense* igo_allocate_dense2 (
    /* --- input --- */
    cholmod_dense** B_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* B = *B_handle;
    igo_dense* igo_B = (igo_dense*) malloc(sizeof(igo_dense));
    igo_B->nzmax_alloc = B->nzmax;
    igo_B->B = B;
    *B_handle = NULL;

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

/* Copys an igo_dense matrix
 * */
igo_dense* igo_copy_dense (
    /* --- input --- */
    igo_dense* B,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* cholmod_B = cholmod_copy_dense(B->B, igo_cm->cholmod_cm);
    return igo_allocate_dense2(&cholmod_B, igo_cm);
}

/* Wrapper around cholmod_zeros
 * */
igo_dense* igo_zeros (
    /* --- input --- */
    size_t nrow,
    size_t ncol,
    int xtype,
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* zeros = cholmod_zeros(nrow, ncol, xtype, igo_cm->cholmod_cm);
    return igo_allocate_dense2(&zeros, igo_cm);
}

igo_dense* igo_dense_submatrix (
    /* --- input --- */
    igo_dense* B,
    int* Rset,
    int Rsize,
    int* Cset,
    int Csize,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_dense* Bsub = igo_allocate_dense(Rsize, Csize, Rsize, igo_cm);

    int Bd = B->B->d;
    int Bsub_d = Bsub->B->d;
    double* Bx = (double*) B->B->x;
    double* Bsub_x = (double*) Bsub->B->x;

    for(int j = 0; j < Csize; j++) {
        double* Bcol = Bx + Cset[j] * Bd;
        for(int i = 0; i < Rsize; i++) {
            Bsub_x[i] = Bcol[Rset[i]];
        }
        Bsub_x += Bsub_d;
    }

    return Bsub;
}

/* Resize an igo_dense B to (nrow, ncol, d)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes */
int igo_resize_dense (
    /* --- input --- */
    int nrow,
    int ncol,
    int d,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) {

    // Error checking
    assert(d >= nrow);

    cholmod_dense* B = igo_B->B;

    int nrow_old = B->nrow;
    int ncol_old = B->ncol;
    int d_old = B->d;
    int nzmax_old = B->nzmax;

    int nzmax = d * ncol;

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
        double* old_col_start = Bx + ncol * d_old;
        double* new_col_start = Bx + ncol * d;
        for(int j = ncol - 1; j >= ncol_old; j--) {
            old_col_start -= d_old;
            new_col_start -= d;
            memset(new_col_start, 0, nrow * sizeof(double));
        }
        for(int j = ncol_old - 1; j >= 0; j--) {
            old_col_start -= d_old;
            new_col_start -= d;
            memmove(new_col_start, old_col_start, nrow_old * sizeof(double));
            memset(new_col_start + nrow_old, 0, (nrow - nrow_old) * sizeof(double));
        }
    }

    return 1;
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
    int nrow_old = B->nrow;
    int ncol_old = B->ncol;
    int d_old = B->d;
    int nrow_new = B->nrow + Bhat->nrow;
    int ncol_new = B->ncol > Bhat->ncol? B->ncol : Bhat->ncol;
    int d_new = B->d >= nrow_new? B->d : nrow_new + igo_cm->DENSE_D_GROWTH; 

    igo_resize_dense(nrow_new, ncol_new, d_new, igo_B, igo_cm);

    double* old_col_start = (double*) Bhat->x;
    double* new_col_start = (double*) B->x + nrow_old;
    for(int j = 0; j < Bhat->ncol; j++) {
        memcpy(new_col_start, old_col_start, Bhat->nrow * sizeof(double));
        old_col_start += Bhat->d;
        new_col_start += B->d;
    }

    return 1;
}

/* Perform igo_B = [igo_B; igo_Bhat]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_vertappend_dense2 (
    /* --- input --- */
    igo_dense* igo_Bhat,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) {
  return igo_vertappend_dense(igo_Bhat->B, igo_B, igo_cm);
}

int igo_vertappend_sparse_to_dense (
    /* --- input --- */
    cholmod_sparse* Bhat, 
    /* --- in/out --- */
    igo_dense* igo_B, 
    /* ------------- */
    igo_common* igo_cm
) {
    cholmod_dense* B = igo_B->B;
    int nrow_old = B->nrow;
    int ncol_old = B->ncol;
    int d_old = B->d;
    int nrow_new = B->nrow + Bhat->nrow;
    int ncol_new = B->ncol > Bhat->ncol? B->ncol : Bhat->ncol;
    int d_new = B->d >= nrow_new? B->d : nrow_new + igo_cm->DENSE_D_GROWTH; 

    igo_resize_dense(nrow_new, ncol_new, d_new, igo_B, igo_cm);

    int* Bhat_p = (int*) Bhat->p;
    int* Bhat_i = (int*) Bhat->i;
    double* Bhat_x = (double*) Bhat->x;
    double* Bcol_start = (double*) B->x;
    for(int j = 0; j < Bhat->ncol; j++) {
        int col_start = Bhat_p[j];
        int col_end = Bhat_p[j + 1];
        for(int idx = col_start; idx < col_end; idx++) {
            int row = Bhat_i[idx];
            Bcol_start[nrow_old + row] = Bhat_x[idx];
        }
        Bcol_start += d_new;
    }

    return 1;
}

int igo_vertappend_sparse_to_dense2 (
    /* --- input --- */
    igo_sparse* igo_Bhat, 
    /* --- in/out --- */
    igo_dense* igo_B, 
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_vertappend_sparse_to_dense(igo_Bhat->A, igo_B, igo_cm);
}

/* Replace the nonzero columns of B with corresponding columns in B_tilde
 * Return the replaced submatrix with the same pattern as B_tilde
 * */
igo_sparse* igo_replace_dense (
    /* --- input --- */
    igo_dense* B,
    igo_sparse* B_tilde,
    /* ------------- */
    igo_common* igo_cm
) {
    assert(B_tilde->A->ncol <= B->B->ncol);
    assert(B_tilde->A->nrow <= B->B->nrow);
    assert(B_tilde->A->packed);

    igo_sparse* B_tilde_neg = igo_copy_sparse(B_tilde, igo_cm);

    int* B_tilde_p = (int*) B_tilde->A->p;
    int* B_tilde_i = (int*) B_tilde->A->i;
    double* B_tilde_x = (double*) B_tilde->A->x;
    double* B_tilde_neg_x = (double*) B_tilde_neg->A->x;
    int B_tilde_ncol = B_tilde->A->ncol;

    double* Bx = (double*) B->B->x;
    double* B_col_x = Bx;
    int Bd = B->B->d;

    for(int j = 0; j < B_tilde_ncol; j++) {
        int col_start = B_tilde_p[j];
        int col_end = B_tilde_p[j + 1];
        int col_nz = col_end - col_start;

        for(int idx = col_start; idx < col_end; idx++) {
            int i = B_tilde_i[idx];
            B_tilde_neg_x[idx] = B_col_x[i];
            B_col_x[i] = B_tilde_x[idx];
        }

        B_col_x += Bd;
    }
    
    return B_tilde_neg;
}

/* Permute rows in a dense matrix
 * Currently used for a dense vector
 * */
int igo_permute_rows_dense (
    /* --- input --- */
    igo_dense* B,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) {
    double* Bcol = (double*) B->B->x;
    int nrow = B->B->nrow;
    int Bd = B->B->d;
    double* Bcol_copy = (double*) malloc(nrow * sizeof(double));
    
    for(int j = 0; j < B->B->ncol; j++) {
        memcpy(Bcol_copy, Bcol, nrow * sizeof(double));
        for(int idx = 0; idx < nrow; idx++) {
            int i = P[idx];
            Bcol[idx] = Bcol_copy[i];
        }
        Bcol += Bd;
    }

    free(Bcol_copy);
    Bcol_copy = NULL;

    return 1;
}

/* Unpermute rows in a dense matrix
 * Currently used for a dense vector
 * */
int igo_unpermute_rows_dense (
    /* --- input --- */
    igo_dense* B,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) {
    double* Bcol = (double*) B->B->x;
    int nrow = B->B->nrow;
    int Bd = B->B->d;
    double* Bcol_copy = (double*) malloc(nrow * sizeof(double));
    
    for(int j = 0; j < B->B->ncol; j++) {
        memcpy(Bcol_copy, Bcol, nrow * sizeof(double));
        for(int idx = 0; idx < nrow; idx++) {
            int i = P[idx];
            Bcol[i] = Bcol_copy[idx];
        }
        Bcol += Bd;
    }

    free(Bcol_copy);
    Bcol_copy = NULL;

    return 1;
}

/* Test if two cholmod_dense B1 and B2 are equal, 
 * i.e.|B1 - B2|_infty < eps
 * */
bool igo_cholmod_dense_eq(
    /* --- input --- */
    cholmod_dense* B1,
    cholmod_dense* B2,
    double eps,
    /* ------------- */
    cholmod_common* igo_cm
) {
    if(B1 == NULL && B2 == NULL) { return true; }
    if(B1 == NULL || B2 == NULL) { return false; }
    if(B1->nrow != B2->nrow) { return false; }
    if(B1->ncol != B2->ncol) { return false; }

    int B1d = B1->d;
    int B2d = B2->d;
    double* B1x = (double*) B1->x;
    double* B2x = (double*) B2->x;
    for(int j = 0; j < B1->ncol; j++) {
        double* B1xcol = B1x;
        double* B2xcol = B2x;
        for(int i = 0; i < B1->nrow; i++) {
            if(fabs(*B1xcol - *B2xcol) >= eps) { return false; }
            B1xcol++;
            B2xcol++;
        }
        B1x += B1d;
        B2x += B2d;
    }

    return true;
}

/* Wrapper around igo_cholmod_dense_eq
 * */
bool igo_dense_eq(
    /* --- input --- */
    igo_dense* B1,
    igo_dense* B2,
    double eps,
    /* ------------- */
    igo_common* igo_cm
) {
    return igo_cholmod_dense_eq(B1->B, B2->B, eps, igo_cm->cholmod_cm);
}

void igo_print_dense(
    /* --- input --- */
    int verbose,
    char* name,
    igo_dense* igo_B,
    igo_common* igo_cm
) {
    if(!igo_B) {
        printf("Dense matrix %s is NULL\n", name);
        return;
    }
    igo_print_cholmod_dense(verbose, name, igo_B->B, igo_cm->cholmod_cm);
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
