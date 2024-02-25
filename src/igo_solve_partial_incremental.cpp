#include <igo.h>

int igo_solve_partial_incremental (
    /* --- input --- */
    igo_sparse* A,
    igo_dense* b,
    /* --- in/out --- */
    igo_factor** L_handle,
    igo_dense** y_handle,
    igo_dense** x_handle,
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    /* ------------- */
    igo_solve_context* cxt,
    igo_common* igo_cm
) {
    return 1;
}
