#include <igo.h>

igo_AT_pattern* igo_allocate_AT_pattern (
    /* --- input --- */
    int max_col,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_AT_pattern* AT = (igo_AT_pattern*) malloc(sizeof(igo_AT_pattern));

    AT->maxcol = max_col;
    AT->ncol = 0;
    AT->maxlen = (int*) malloc(AT->maxcol * sizeof(int));
    AT->len = (int*) malloc(AT->maxcol * sizeof(int));

    memset(AT->len, 0, AT->maxcol * sizeof(int));

    AT->i = (int**) malloc(AT->maxcol * sizeof(int*));

    for(int j = 0; j < AT->maxcol; j++) {
        AT->maxlen[j] = IGO_SPARSE_DEFAULT_NROW_ALLOC;
        AT->i[j] = (int*) malloc(AT->maxlen[j] * sizeof(int));
        AT->len[j] = 0;
    }

    return AT;
}

void igo_free_AT_pattern (
    /* --- input --- */
    igo_AT_pattern** AT_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    if(!AT_handle) { return; }

    igo_AT_pattern* AT = *AT_handle;

    if(!AT) { return; }

    for(int j = 0; j < AT->maxcol; j++) {
        free(AT->i[j]);
    }

    free(AT->i);
    free(AT->len);
    free(AT->maxlen);

    free(AT);
    *AT_handle = NULL;

    return;
}

void igo_resize_AT_pattern (
    /* --- input --- */
    int ncol,
    /* --- in/out --- */
    igo_AT_pattern* AT,
    /* ------------- */
    igo_common* igo_cm
) {
    if(ncol > AT->maxcol) {
        int oldmaxcol = AT->maxcol;

        do {
            AT->maxcol *= 2;
        } while(ncol > AT->maxcol);

        AT->maxlen = (int*) realloc(AT->maxlen, AT->maxcol * sizeof(int));
        AT->len = (int*) realloc(AT->len, AT->maxcol * sizeof(int));
        AT->i = (int**) realloc(AT->i, AT->maxcol * sizeof(int*));

        for(int j = oldmaxcol; j < AT->maxcol; j++) {
            AT->maxlen[j] = IGO_SPARSE_DEFAULT_NROW_ALLOC;
            AT->i[j] = (int*) malloc(AT->maxlen[j] * sizeof(int));
            AT->len[j] = 0;
        }
    }

    AT->ncol = ncol;
}

void igo_AT_col_pushback (
    /* --- input --- */
    int col,
    int val,
    /* --- in/out --- */
    igo_AT_pattern* AT,
    /* ------------- */
    igo_common* igo_cm
) {
    if(AT->len[col] >= AT->maxlen[col]) {
        AT->maxlen[col] *= 2;

        AT->i[col] = (int*) realloc(AT->i[col], AT->maxlen[col] * sizeof(int));
    }

    AT->i[col][AT->len[col]] = val;
    AT->len[col]++;
}

void igo_AT_append_A_hat (
    /* --- input --- */
    int orig_cols,
    igo_sparse* A_hat,
    /* --- in/out --- */
    igo_AT_pattern* AT,
    /* ------------- */
    igo_common* igo_cm
) {
    int nrow = A_hat->A->nrow;
    int ncol = A_hat->A->ncol;
    int* Ap = (int*) A_hat->A->p;
    int* Ai = (int*) A_hat->A->i;

    igo_resize_AT_pattern(nrow, AT, igo_cm);

    for(int j = 0; j < ncol; j++) {
        for(int idx = Ap[j]; idx < Ap[j + 1]; idx++) {
            int row = Ai[idx];

            igo_AT_col_pushback(row, orig_cols + j, AT, igo_cm);
        }
    }

}
