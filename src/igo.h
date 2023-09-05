#ifndef IGO_H
#define IGO_H

#include <cholmod.h>

/* ---------------------------------------------------------- */
/* Object definitions */
/* ---------------------------------------------------------- */

/* Wrapper around cholmod_sparse for better memory management to support growing matrix */
typedef struct igo_sparse_struct {

    cholmod_sparse* A;
    /* How much is actually allocated so we can amortize the allocation cost. */
    int ncol_alloc;
    int nzmax_alloc;  

} igo_sparse ;

/* Wrapper around cholmod_dense for better memory management to support growing matrix */
typedef struct igo_dense_struct {

    cholmod_dense* B;
    /* How much is actually allocated so we can amortize the allocation cost. */
    int nzmax_alloc;  

} igo_dense ;

/* Wrapper around cholmod_factor for better memory management to support growing matrix */
typedef struct igo_factor_struct {

    cholmod_factor* L;
    /* How much is actually allocated so we can amortize the allocation cost. */
    int n_alloc;
    int nzmax_alloc;  

} igo_factor ;

typedef struct igo_common_struct {

    igo_sparse* A;

    igo_dense* delta_Atb;

    igo_factor* L;

    igo_dense* x;

    igo_dense* y;   // y = L^(-1) Atb

    cholmod_common* cholmod_cm;

} igo_common ;

/* ---------------------------------------------------------- */
/* Primary functions */
/* ---------------------------------------------------------- */

int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) ;

int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) ;

int igo_solve_increment (
    /* --- inputs --- */   
    cholmod_sparse* A_tilde, 
    cholmod_dense* b_tilde,
    cholmod_sparse* A_hat,
    cholmod_dense* b_hat,
    /* --- outputs --- */
    cholmod_dense* x,
    /* --- common --- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Sparse matrix functions */
/* ---------------------------------------------------------- */

/* Initialize an igo_sparse_matrix */
igo_sparse* igo_allocate_sparse (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_sparse (
    /* --- in/out --- */
    igo_sparse** A,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

/* Perform igo_cm->A = [igo_cm->A B]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_horzappend_sparse (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Dense matrix functions */
/* ---------------------------------------------------------- */

/* Initialize an igo_dense matrix */
igo_dense* igo_allocate_dense (
    /* --- input --- */
    int nrow,
    int ncol,
    int d,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_dense (
    /* --- in/out --- */
    igo_dense** igo_B_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

/* Perform [igo_B->B; Bhat] = [igo_B->B; Bhat]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_vertappend_dense (
    /* --- input --- */
    cholmod_dense* Bhat,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Sparse factor functions */
/* ---------------------------------------------------------- */

/* Initialize an igo_factor */
igo_factor* igo_allocate_factor (
    /* --- input --- */
    size_t n,
    size_t nzmax,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_factor (
    /* --- in/out --- */
    igo_factor** L_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Resize an igo_sparse A to (nrow, ncol, nzmax)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes */
int igo_resize_factor (
    /* --- input --- */
    size_t n,
    size_t nzmax,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_updown (
    /* --- input --- */
    int update,             // 1 for update, 0 for downdate
    cholmod_sparse* Ahat,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_updown_solve (
    /* --- input --- */
    int update,             // 1 for update, 0 for downdate
    cholmod_sparse* Ahat,
    /* --- in/out --- */
    igo_factor* igo_L,
    igo_dense* igo_x,
    igo_dense* igo_delta_b,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Print functions */
/* ---------------------------------------------------------- */

void igo_print_cholmod_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_sparse* A,
    cholmod_common* cholmod_cm
) ;

void igo_print_cholmod_dense(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_dense* B,
    cholmod_common* cholmod_cm
) ;

void igo_print_cholmod_factor(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_factor* L,
    cholmod_common* cholmod_cm
) ;

void igo_print_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    igo_sparse* igo_A,
    igo_common* igo_cm
) ;

void igo_print_dense(
    /* --- input --- */
    int verbose,
    char* name,
    igo_dense* igo_B,
    igo_common* igo_cm
) ;

void igo_print_factor(
    /* --- input --- */
    int verbose,
    char* name,
    igo_factor* L,
    igo_common* igo_cm
) ;

#endif  // IGO_H
