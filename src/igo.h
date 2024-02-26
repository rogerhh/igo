#ifndef IGO_H
#define IGO_H

#include <cholmod.h>

/* ---------------------------------------------------------- */
/* Object definitions */
/* ---------------------------------------------------------- */

#define IGO_SPARSE_DEFAULT_NCOL_ALLOC 32
#define IGO_SPARSE_DEFAULT_NZMAX_ALLOC 64

#define IGO_FACTOR_DEFAULT_COL_SIZE 32

#define IGO_PERM_DEFAULT_N_ALLOC IGO_SPARSE_DEFAULT_NCOL_ALLOC

#define IGO_DEFAULT_BATCH_SOLVE_THRESH 0.5

#define IGO_REORDER_PERIOD 100

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

/* Wrapper around cholmod_triplet for better memory management to support growing matrix */
typedef struct igo_triplet_struct {

    cholmod_triplet* A;
    /* How much is actually allocated so we can amortize the allocation cost. */
    int ncol_alloc;
    int nzmax_alloc;  

} igo_triplet ;

/* Wrapper around cholmod_factor for better memory management to support growing matrix */
typedef struct igo_factor_struct {

    cholmod_factor* L;
    /* How much is actually allocated so we can amortize the allocation cost. */
    int n_alloc;

    /* We do not need nzmax_alloc here as L->nzmax already keeps track */

    // /* We should make sure that nzmax_alloc is the same as L->nzmax, because 
    //  * cholmod_updown* realloc's memory, so we need to update nzmax_alloc whenever
    //  * we call cholmod_updown* */
    // int nzmax_alloc;  

} igo_factor ;

typedef struct igo_perm_struct {
    int* P;
    int n_alloc;
    int n;
} igo_perm ;

typedef struct igo_common_struct {

    igo_sparse* A;

    igo_dense* b;

    igo_factor* L;

    // igo_dense* Ab;

    igo_dense* x;

    igo_dense* y;   // y = L^(-1) Atb

    int reorder_counter;

    cholmod_common* cholmod_cm;

    int FACTOR_NCOL_ALLOC;
    int FACTOR_NZMAX_ALLOC;
    int FACTOR_DEFAULT_COL_SIZE;

    int DENSE_D_GROWTH;

    double BATCH_SOLVE_THRESH;

    int REORDER_PERIOD;

} igo_common ;

/* ---------------------------------------------------------- */
/* Primary functions */
/* ---------------------------------------------------------- */

/* Returns 0 if given a NULL pointer, 1 otherwise. */
int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) ;

/* Always succeeds and returns 1. Attempting to finish a NULL pointer performs no operation on it instead. */
int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) ;

int igo_solve_increment (
    /* --- inputs --- */   
    igo_sparse* A_tilde, 
    igo_sparse* b_tilde,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- outputs --- */
    // igo_dense* x,
    /* --- common --- */
    igo_common* igo_cm
) ;

int igo_solve_increment2 (
    /* --- inputs --- */   
    igo_sparse* A_tilde, 
    igo_sparse* b_tilde,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- outputs --- */
    // igo_dense* x,
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

/* Initialize an igo_sparse matrix with an existing cholmod_sparse_matrix 
 * Destroys the original sparse matrix pointer */
igo_sparse* igo_allocate_sparse2 (
    /* --- input --- */
    cholmod_sparse** A_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_sparse (
    /* --- in/out --- */
    igo_sparse** A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Copys an igo_sparse matrix
 * */
igo_sparse* igo_copy_sparse (
    /* --- input --- */
    igo_sparse* A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Resize an igo_sparse A to (nrow, ncol, nzmax)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes
 * Requires: nrow >= igo_A->A->nrow
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
) ;

/* Perform igo_A = [igo_A B]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_horzappend_sparse (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Performs igo_A = [igo_A; B]. 
 * This is needed because cholmod_vertcat makes copies of the inputs */
int igo_vertappend_sparse (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Performs igo_A = [igo_A igo_B]. 
 * This is needed because cholmod_vertcat makes copies of the inputs */
int igo_horzappend_sparse2 (
    /* --- input --- */
    igo_sparse* igo_B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Performs igo_A = [igo_A; igo_B]
 * This is needed because cholmod_vertcat makes copies of the inputs */
int igo_vertappend_sparse2 (
    /* --- input --- */
    igo_sparse* igo_B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Count number of nonzero columns
 * */
int igo_count_nz_cols (
    /* --- input --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Drop unused columns
 * */
int igo_drop_cols_sparse (
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Drop unused rows
 * */
int igo_drop_rows_sparse (
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Wrapper around cholmod_ssmult
 * */
igo_sparse* igo_ssmult (
    /* --- input --- */
    igo_sparse* igo_A,
    igo_sparse* igo_B,
    int stype,
    int values,
    int sorted,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

/* Replace the nonzero columns of A with corresponding columns in A_tilde
 * Return the replaced submatrix in the same pattern as A_tilde
 * */
igo_sparse* igo_replace_sparse (
    /* --- input --- */
    igo_sparse* igo_A,
    igo_sparse* A_tilde,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

igo_sparse* igo_submatrix2 (
    /* --- input --- */
    igo_sparse* A,
    igo_perm* row_perm,
    igo_perm* col_perm,
    int values,
    int sorted,
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

/* Initialize an igo_dense matrix with and existing dense matrix
 * Destroys the original dense matrix pointer */
igo_dense* igo_allocate_dense2 (
    /* --- input --- */
    cholmod_dense** B_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_dense (
    /* --- in/out --- */
    igo_dense** igo_B_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Copys an igo_dense matrix
 * */
igo_dense* igo_copy_dense (
    /* --- input --- */
    igo_dense* B,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Wrapper around cholmod_zeros
 * */
igo_dense* igo_zeros (
    /* --- input --- */
    size_t nrow,
    size_t ncol,
    int xtype,
    /* ------------- */
    igo_common* igo_cm
) ;

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

/* Perform igo_B->B = [igo_B; igo_Bhat]. 
 * This is needed because cholmod_horzcat makes copies of the inputs */
int igo_vertappend_dense2 (
    /* --- input --- */
    igo_dense* igo_Bhat,
    /* --- in/out --- */
    igo_dense* igo_B,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_vertappend_sparse_to_dense (
    /* --- input --- */
    cholmod_sparse* Bhat, 
    /* --- in/out --- */
    igo_dense* igo_B, 
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_vertappend_sparse_to_dense2 (
    /* --- input --- */
    igo_sparse* igo_Bhat, 
    /* --- in/out --- */
    igo_dense* igo_B, 
    /* ------------- */
    igo_common* igo_cm
) ;

/* Replace the nonzero columns of B with corresponding columns in B_tilde
 * Return the replaced submatrix with the same pattern as B_tilde
 * */
igo_sparse* igo_replace_dense (
    /* --- input --- */
    igo_dense* B,
    igo_sparse* B_tilde,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Permute rows in a dense matrix
 * Currently used for a dense vector
 * */
int igo_permute_rows_dense (
    /* --- input --- */
    igo_dense* B,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Unpermute rows in a dense matrix
 * Currently used for a dense vector
 * */
int igo_unpermute_rows_dense (
    /* --- input --- */
    igo_dense* B,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

/* Wrapper around igo_cholmod_dense_eq
 * */
bool igo_dense_eq(
    /* --- input --- */
    igo_dense* B1,
    igo_dense* B2,
    double eps,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Triplet matrix functions */
/* ---------------------------------------------------------- */

/* Initialize an igo_sparse_matrix */
igo_sparse* igo_allocate_triplet (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_triplet (
    /* --- in/out --- */
    igo_triplet** T,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Resize an igo_sparse A to (nrow, ncol, nzmax)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes */
int igo_resize_triplet (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    /* --- in/out --- */
    igo_triplet* igo_T,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Sparse factor functions */
/* ---------------------------------------------------------- */

/* Initialize an igo_factor */
igo_factor* igo_allocate_factor (
    /* --- input --- */
    int n,
    int nzmax,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Initialize an igo_factor from an existing cholmod_factor
 * Destroys the original factor pointer */
igo_factor* igo_allocate_factor2 (
    /* --- input --- */
    cholmod_factor** L_handle,
    /* ------------- */
    igo_common* igo_cm
) ;


void igo_free_factor (
    /* --- in/out --- */
    igo_factor** L_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Copys an igo_factor
 * */
igo_factor* igo_copy_factor (
    /* --- input --- */
    igo_factor* L,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Resize an igo_factor A to (nrow, ncol, nzmax)
 * The actual underlying memory might be larger than specified
 * to accomodate for future resizes.
 * nzmax is applied to the old columns of the factor,
 * the new columns will have igo_cm->FACTOR_DEFAULT_COL_SIZE nonzeros*/
int igo_resize_factor (
    /* --- input --- */
    int n,
    int nzmax,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

/* After updown solve, the nzmax of a factor igo_L->L may be larger than 
 * igo_L->nzmax_alloc. This happens when cholmod_updown_solve reallocates 
 * memory. In this case, resize igo_L to the next power of 2 to 
 * accomodate for future growth
 * */
int igo_factor_adjust_nzmax (
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Combine cholmod_analyze and cholmod_factorize in one step
 * Takes in a sparse matrix A 
 * */
igo_factor* igo_analyze_and_factorize (
    /* --- input --- */
    igo_sparse* A,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_updown (
    /* --- input --- */
    int update,             // 1 for update, 0 for downdate
    igo_sparse* igo_A,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Wrapper around cholmod_updown2.
 * Computes the new Cholesky factor of LL' + CC' - DD'
 * */
int igo_updown2 (
    /* --- input --- */
    igo_sparse* igo_C,
    igo_sparse* igo_D,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

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
) ;

/* Wrapper around cholmod_updown2_solve
 * Computes the new Cholesky factor of LL' + CC' - DD'
 * */
int igo_updown2_solve (
    /* --- input --- */
    igo_sparse* igo_C,
    igo_sparse* igo_D,
    /* --- in/out --- */
    igo_factor* igo_L,
    igo_dense* igo_x,
    igo_dense* igo_delta_b,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Wrapper around cholmod_solve
 * */
igo_dense* igo_solve (
    /* --- input --- */
    int sys,            // System to solve
    igo_factor* igo_L,  // Cholesky factorization
    igo_dense* igo_B,   // Right hand side matrix
    /* ------------- */
    igo_common* igo_cm
) ;

/* Test if two cholmod_factors L1 and L2 are equal, 
 * i.e. nonzero(L1) == nonzero(L2) and |L1 - L2|_infty < eps
 * */
bool igo_cholmod_factor_eq(
    /* --- input --- */
    cholmod_factor* L1,
    cholmod_factor* L2,
    double eps,
    /* ------------- */
    cholmod_common* cholmod_cm
) ;

/* Wrapper around igo_cholmod_factor_eq
 * */
bool igo_factor_eq(
    /* --- input --- */
    igo_factor* L1,
    igo_factor* L2,
    double eps,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Permutation functions */
/* ---------------------------------------------------------- */

igo_perm* igo_allocate_perm (
    /* --- input --- */
    int len,
    /* ------------- */
    igo_common* igo_cm
) ;

igo_perm* igo_allocate_perm2 (
    /* --- input --- */
    int len,
    int** P,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_perm (
    /* --- in/out --- */
    igo_perm** P_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Perform P2 = P2 * P1
 * */
int igo_permute_permutation (
    /* --- input --- */
    int len,
    int* P1,
    /* --- in/out --- */
    int* P2,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Return P^(-1)
 * */
int* igo_invert_permutation (
    /* --- input --- */
    int len,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Extend permutation to newlen
 * */
int igo_extend_permutation (
    /* --- input --- */
    int Plen,
    int newlen,
    /* --- in/out --- */
    int* P,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Extend permutation to newlen
 * */
int igo_extend_permutation2 (
    /* --- input --- */
    int newlen,
    /* --- in/out --- */
    igo_perm* P,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_print_permutation (
    /* --- input --- */
    int len,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_print_permutation2 (
    /* --- input --- */
    igo_perm* P,
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
