#include <igo.h>

igo_solve_context* igo_allocate_solve_context (
    /* ------------- */
    igo_common* igo_cm
) {
    igo_solve_context* cxt = (igo_solve_context*) malloc(sizeof(igo_solve_context));

    cxt->A_tilde_nz_cols = 0;
    cxt->A_hat_nz_cols = 0;
    cxt->changed_cols = 0;
    cxt->orig_cols = 0;
    cxt->new_cols = 0;
    cxt->num_affected_cols = 0;

    cxt->num_relin_staged_cols = 0;

    cxt->h_orig = 0;
    cxt->h_hat = 0;
    cxt->h_tilde = 0;

    cxt->PA = NULL;
    cxt->PAb = NULL;

    cxt->A_tilde_neg = NULL;
    cxt->b_tilde_neg = NULL;

    cxt->PA_sel = NULL;
    cxt->PA_sel_neg = NULL;

    cxt->PA_hat = NULL;
    cxt->PAb_hat = NULL;

    cxt->PAb_delta = NULL;

    cxt->num_affected_rows = 0;
    cxt->affected_rows = NULL;
    cxt->row_map = NULL;
    cxt->L_map = NULL;
    cxt->L_map_inv = NULL;

    cxt->num_L21_cols = 0;
    cxt->L21_cols = NULL;

    cxt->num_affected_cols = 0;
    cxt->affected_cols = NULL;

    cxt->num_sel_cols = 0;
    cxt->sel_cols = NULL;

    cxt->pcg_cxt = NULL;

    return cxt;
}

void igo_free_solve_context (
    /* --- in/out --- */
    igo_solve_context** cxt_handle,
    /* ------------- */
    igo_common* igo_cm
) {
    if(!cxt_handle) { return; }
    igo_solve_context* cxt = *cxt_handle;

    igo_free_sparse(&cxt->PA, igo_cm);
    igo_free_dense(&cxt->PAb, igo_cm);

    igo_free_sparse(&cxt->A_tilde_neg, igo_cm);
    igo_free_sparse(&cxt->b_tilde_neg, igo_cm);

    igo_free_sparse(&cxt->PA_hat, igo_cm);
    igo_free_sparse(&cxt->PAb_hat, igo_cm);

    igo_free_sparse(&cxt->PA_sel, igo_cm);
    igo_free_sparse(&cxt->PA_sel_neg, igo_cm);

    igo_free_dense(&cxt->PAb_delta, igo_cm);

    free(cxt->affected_rows);
    free(cxt->row_map);
    free(cxt->L_map);
    free(cxt->L_map_inv);

    free(cxt->L21_cols);

    free(cxt->affected_cols);

    free(cxt->sel_cols);

    free(cxt->pcg_cxt);

    free(*cxt_handle);
    *cxt_handle = NULL;
    return;
}
