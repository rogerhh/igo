#ifndef IGO_H
#define IGO_H

extern "C" {
#include <cholmod.h>
}

/* ---------------------------------------------------------- */
/* Object definitions */
/* ---------------------------------------------------------- */

#define IGO_SPARSE_DEFAULT_NROW_ALLOC 32
#define IGO_SPARSE_DEFAULT_NCOL_ALLOC 32
#define IGO_SPARSE_DEFAULT_NZMAX_ALLOC 64

#define IGO_FACTOR_DEFAULT_COL_SIZE 32

#define IGO_FACTOR_DEFAULT_SUBFACTOR_GROW 1.2

#define IGO_PERM_DEFAULT_N_ALLOC IGO_SPARSE_DEFAULT_NCOL_ALLOC

#define IGO_VECTOR_DEFAULT_SIZE 32

#define IGO_DEFAULT_BATCH_SOLVE_THRESH 0.5

#define IGO_REORDER_PERIOD 100

#define IGO_SOLVE_PARTIAL_DECIDE -1
#define IGO_SOLVE_PARTIAL_FALSE 0
#define IGO_SOLVE_PARTIAL_TRUE 1
#define IGO_DEFAULT_PARTIAL_THRESH 0.65 

#define IGO_DEFAULT_SEL_COLS_RATE 0.05
#define IGO_DEFAULT_MIN_SEL_COLS 128
#define IGO_DEFAULT_PCG_RTOL 1e-10
#define IGO_DEFAULT_PCG_ATOL 1e-10

#define IGO_SOLVE_DECIDE      -1
#define IGO_SOLVE_BATCH       0
#define IGO_SOLVE_INCREMENTAL 1
#define IGO_SOLVE_PCG         2

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

typedef struct igo_pcg_context_struct {
    int num_iter;
    double aerr;
    double rerr;
} igo_pcg_context;

typedef struct igo_solve_context_struct {
    // All dynamically allocated variables are defined here 
    // so they can be shared and deallocated at the same time

    int A_tilde_nz_cols;
    int A_hat_nz_cols;
    int changed_cols;
    int orig_cols;
    int new_cols;

    int num_relin_staged_cols;

    int h_orig;
    int h_hat;
    int h_tilde;

    igo_sparse* PA;
    igo_dense* PAb;

    igo_sparse* A_tilde_neg;
    igo_sparse* b_tilde_neg;

    igo_sparse* PA_hat;
    igo_sparse* PAb_hat;

    igo_sparse* PA_sel;
    igo_sparse* PA_sel_neg;

    igo_dense* PAb_delta;


    int num_affected_rows;
    int* affected_rows;
    int* row_map;
    int* L_map;
    int* L_map_inv;

    int num_L21_cols;
    int* L21_cols;

    int num_affected_cols;
    int* affected_cols;

    int num_sel_cols;
    int* sel_cols;

    igo_pcg_context* pcg_cxt;

} igo_solve_context;

#define DEFINE_VECTOR_TYPE(TYPE) \
    typedef struct {             \
        int maxlen;              \
        int len;                 \
        TYPE* data;              \
    } igo_vector_ ## TYPE ;                          

DEFINE_VECTOR_TYPE(int);
DEFINE_VECTOR_TYPE(double);

// typedef struct igo_vector_int_struct {
//     int maxlen;
//     int len;
//     int* data;
// } igo_vector_int;

// Convenience structure for storing the pattern of AT
typedef struct igo_AT_pattern_struct {
                    
    // Store the patten of the transpose so we can index into which rows correspond to 
    // which columns easily
    int maxcol;
    int ncol;
    int* maxlen;
    int* len;
    int** i;

} igo_AT_pattern ;

typedef struct igo_common_struct {

    igo_sparse* A;            // A holds the true coefficient matrix
    igo_sparse* A_staged_neg; // A_staged_neg holds the old columns of A that need to be replaced
    igo_vector_double* A_staged_diff; // Stores a vector of the difference between A and A_staged_neg

    int num_staged_cols;

    igo_dense* b;
    igo_dense* b_staged_neg;  // b_staged_neg holds the old values of b that is not computed in updown2 solve. This is needed because updown2_solve only considers the rhs of the columns of A that changed

    igo_factor* L;

    // igo_dense* Ab;

    igo_dense* x;

    igo_dense* y;   // y = L^(-1) Atb
    
    igo_AT_pattern* AT;

    igo_sparse* H;  // AAt
    igo_dense* w;   // Ab

    int reorder_counter;

    cholmod_common* cholmod_cm;

    int FACTOR_NCOL_ALLOC;
    int FACTOR_NZMAX_ALLOC;
    int FACTOR_DEFAULT_COL_SIZE;

    int DENSE_D_GROWTH;

    double BATCH_SOLVE_THRESH;

    int REORDER_PERIOD;

    int solve_partial;
    double partial_thresh;

    double subfactor_grow;

    double SEL_COLS_RATE;
    int MIN_SEL_COLS;
    double pcg_rtol;
    double pcg_atol;

    int solve_type;

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

/* Solve a subproblem incrementally. 
 * The solved problem is defined as follows.
 * On input
 * Let S be the column selection matrix for the columns where A_staged_diff != 0
 * Let S' be the selection matrix for columns where A_staged_diff = 0
 * Then, 
 * LL^T = (AS')(AS')^T - (A_staged_negS)(A_staged_negS)^T - A_neg A_neg^T
 * Ly = AS' S'^T b - A_staged_neg S S^T b_staged_neg - A_neg A_neg^T b_neg
 * On return,
 * LL^T = (AS')(AS')^T - (A_staged_negS)(A_staged_negS)^T - A_neg A_neg^T
 * Ly = AS' S'^T b - A_staged_neg S S^T b_staged_neg - A_neg A_neg^T b_neg
 * but for adjusted S and S'
 * (AA^T - A_neg A_neg^T) x = Ab - A_neg b_neg
 * Pass handles for parameters that may be re-allocated */
int igo_solve_increment3 (
    /* --- input --- */   
    igo_sparse* A,
    igo_dense* b,
    igo_sparse* PA_neg,
    igo_dense* b_neg,
    int A_hat_col_start,    // All columns after this column is part of A_hat
    int solve_type,
    /* --- in/out --- */   
    igo_sparse* A_staged_neg,
    igo_dense* b_staged_neg,
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    igo_factor** L_handle,
    igo_dense** y_handle,
    /* --- output --- */
    igo_dense** x_handle,
    /* --- common --- */
    igo_common* igo_cm
) ;

// Rewrite of igo_solve_increment2 to make things cleaner
int igo_solve_increment4 (
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

/* Initialize an igo_sparse matrix with all the options provided for cholmod_sparse_matrix */
igo_sparse* igo_allocate_sparse3 (
    /* --- input --- */
    int nrow,
    int ncol,
    int nzmax,
    int sorted,
    int packed,
    int stype,
    int xtype,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_sparse (
    /* --- in/out --- */
    igo_sparse** A,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_check_invariant_sparse (
    /* --- input --- */
    igo_sparse* A,
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

/* Copys an igo_sparse matrix pattern. Do not allocate extra memory
 * */
igo_sparse* igo_copy_sparse_pattern (
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

/* Same as igo_horzappend_sparse, but set the appended entries to 0. */
int igo_horzappend_sparse_pattern (
    /* --- input --- */
    cholmod_sparse* B,
    /* --- in/out --- */
    igo_sparse* igo_A,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Same as igo_horzappend_sparse2, but set the appended entries to 0 */
int igo_horzappend_sparse2_pattern (
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

/* Wrapper around cholmod_aat
 * */
igo_sparse* igo_aat (
    /* --- input --- */
    igo_sparse* igo_A,
    int* fset,
    int fsize,
    int mode,
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

/* Wrapper around cholmod_add
 * */
igo_sparse* igo_add (
    /* --- input --- */
    igo_sparse* igo_A,
    igo_sparse* igo_B,
    double* alpha,
    double* beta,
    int values,
    int sorted,
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
/* AT pattern functions */
/* ---------------------------------------------------------- */

igo_AT_pattern* igo_allocate_AT_pattern (
    /* --- input --- */
    int num_col,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_AT_pattern (
    /* --- input --- */
    igo_AT_pattern** AT_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_resize_AT_pattern (
    /* --- input --- */
    int ncol,
    /* --- in/out --- */
    igo_AT_pattern* AT,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_AT_col_pushback (
    /* --- input --- */
    int col,
    int val,
    /* --- in/out --- */
    igo_AT_pattern* AT,
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_AT_append_A_hat (
    /* --- input --- */
    int orig_cols,
    igo_sparse* A_hat,
    /* --- in/out --- */
    igo_AT_pattern* AT,
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

igo_dense* igo_dense_submatrix (
    /* --- input --- */
    igo_dense* B,
    int* Rset,
    int Rsize,
    int* Cset,
    int Csize,
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

/* Initialize an igo_factor that only has 1s on the diagonal */
igo_factor* igo_allocate_identity_factor (
    /* --- input --- */
    int n,
    int nzmax,
    double d,
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
 * the new columns will have igo_cm->FACTOR_DEFAULT_COL_SIZE nonzeros
 * The new factor will have 1e-12 on the diagonal */
int igo_resize_factor (
    /* --- input --- */
    int n,
    int nzmax,
    /* --- in/out --- */
    igo_factor* igo_L,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Resize an igo_factor A to (nrow, ncol, nzmax) 
 * Same as igo_resize_factor but the new factor will have d on the diagonal */
int igo_resize_factor2 (
    /* --- input --- */
    int n,
    int nzmax,
    double d,
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

/* Combine cholmod_analyze_p and cholmod_factorize in one step
 * Takes in a sparse matrix PA 
 * */
igo_factor* igo_analyze_p_and_factorize (
    /* --- input --- */
    igo_sparse* A,
    int32_t *UserPerm,	/* user-provided permutation, size A->nrow */
    int32_t *fset,	/* subset of 0:(A->ncol)-1 */
    size_t fsize,	/* size of fset */
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

int igo_get_affected_rows (
    /* --- input --- */
    igo_sparse* A_hat,
    igo_sparse* A_staged_neg,
    igo_vector_double* A_staged_diff,
    int ncol,                   // ncol is the number of coumns in A_staged not including A_hat
    igo_factor* L,
    /* --- output --- */
    int* num_affected_rows,
    int* affected_rows,         // size L->n, maps new unpermuted row to old unpermuted row   
    int* row_map,               // size L->n, maps old unpermuted row to new unpermuted row
    int* L_map,                 // size L->n, maps old L rows to new L rows
    int* L_map_inv,             // size L->n, maps new L rows to old L rows
    /* ------------- */
    igo_common* igo_cm
) ;

igo_factor* igo_subfactor (
    /* --- input --- */
    igo_factor* L,
    int num_affected_rows,
    int* affected_rows, // maps new A rows to old A rows
    int* row_map,       // maps old A rows to new A rows
    int* L_map,         // maps old L rows to new L rows
    int* L_map_inv,     // maps new L rows to old L rows
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_get_neg_factor (
    /* --- input --- */
    igo_factor* L,
    igo_dense* y,
    int num_affected_rows,
    int* affected_rows, // maps new A rows to old A rows
    int* row_map,       // maps old A rows to new A rows
    int* L_map,         // maps old L rows to new L rows
    int* L_map_inv,     // maps new L rows to old L rows
    /* --- output --- */
    int* num_L21_cols,
    int* L21_cols,
    igo_sparse* PA_neg,
    igo_dense* b_neg,
    /* ------------- */
    igo_common* igo_cm
) ;

int igo_get_affected_rows (
    /* --- input --- */
    igo_sparse* A_hat,
    igo_sparse* A_staged_neg,
    igo_vector_double* A_staged_diff,
    int ncol,                   // ncol is the number of coumns in A_staged not including A_hat
    igo_factor* L,
    /* --- output --- */
    int* num_affected_rows,
    int* affected_rows,         // size L->n, maps new unpermuted row to old unpermuted row   
    int* row_map,               // size L->n, maps old unpermuted row to new unpermuted row
    int* L_map,                 // size L->n, maps old L rows to new L rows
    int* L_map_inv,             // size L->n, maps new L rows to old L rows
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
/* Solve context functions */
/* ---------------------------------------------------------- */

igo_solve_context* igo_allocate_solve_context (
    /* ------------- */
    igo_common* igo_cm
) ;

void igo_free_solve_context (
    /* --- in/out --- */
    igo_solve_context** cxt_handle,
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Solver functions */
/* ---------------------------------------------------------- */

int igo_solve_full_batch (
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
) ;

int igo_solve_full_incremental (
    /* --- input --- */
    igo_sparse* A,
    igo_dense* b,
    igo_sparse* A_tilde,
    igo_sparse* b_tilde,
    igo_sparse* A_tilde_neg,
    igo_sparse* b_tilde_neg,
    igo_sparse* A_hat,
    igo_sparse* b_hat,
    /* --- in/out --- */
    igo_sparse* A_staged_neg,
    igo_dense* b_staged_neg,
    igo_factor** L_handle,
    igo_dense** y_handle,
    igo_dense** x_handle,
    int* num_staged_cols,
    igo_vector_double* A_staged_diff,
    /* ------------- */
    igo_solve_context* cxt,
    igo_common* igo_cm
) ;

int igo_solve_partial_batch (
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
) ;

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
) ;

/* Solve (AA^T - A_negA_neg^T)x = Ab
 * AA^T - A_negA_neg^T must be SPD
 * M is the preconditioner
 * x0 is the initial guess
 * */
int igo_solve_pcgne(
    /* --- input --- */
    igo_sparse* A, 
    igo_sparse* A_neg, 
    igo_dense* b, 
    igo_factor* M,
    double rtol,
    double atol,
    int max_iter,
    /* --- output --- */
    igo_dense* x, 
    igo_pcg_context* cxt,   // Context for meta data
    /* ------------- */
    igo_common* igo_cm
) ;

/* ---------------------------------------------------------- */
/* Solver util functions */
/* ---------------------------------------------------------- */

int igo_replace_staged (
    /* --- input --- */
    igo_sparse* A_tilde, 
    igo_sparse* A_tilde_neg, 
    igo_sparse* A_staged_neg, 
    igo_sparse* b_tilde,
    igo_sparse* b_tilde_neg,
    igo_dense* b_staged_neg,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* ------------- */
    igo_common* igo_cm
) ;

/* Return a list of indices to at most max_k columns with the highest 
 * difference in A and A_staged_neg
 * Assume indices is already allocated
 * Only consider the first ncol columns. The columns after the first ncol are guaranteed to be picked */
int igo_pick_k_highest_diff (
    /* --- inputs --- */
    int max_k,
    int ncol,
    igo_vector_double* A_staged_diff,
    int num_staged_cols,
    /* --- outputs --- */
    int* k,
    int* indices,
    /* --- common --- */
    igo_common* igo_cm
) ;

igo_dense* igo_compute_PAb_delta_sel(
    /* --- input --- */
    igo_sparse* PA_sel,
    igo_sparse* PA_sel_neg,
    igo_dense* b,
    igo_dense* b_staged_neg,
    int* sel_cols,
    int num_sel_cols,
    /* --- common --- */
    igo_common* igo_cm
) ;

// Set the column nz of the selected columns 0
int igo_set_col_zero(
    /* --- inputs --- */
    int* col_indices,
    int len,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) ;

// Set A_staged_diff = 0 of all columns after A_hat_col_start
int igo_set_A_hat_col_zero(
    /* --- inputs --- */
    int A_hat_col_start,
    /* --- in/out --- */
    igo_vector_double* A_staged_diff,
    int* num_staged_cols,
    /* --- common --- */
    igo_common* igo_cm
) ;

// Assume both inputs are vectors
// Do y += alpha Px 
int igo_add_sparse_to_dense(
    /* --- inputs --- */
    igo_sparse* x,
    double alpha,
    /* --- in/out --- */
    igo_dense* y,
    /* --- common --- */
    igo_common* igo_cm
) ;



/* ---------------------------------------------------------- */
/* Vector functions */
/* ---------------------------------------------------------- */

#define DEFINE_VECTOR_FUNCTIONS(TYPE) \
    igo_vector_##TYPE* igo_allocate_vector_##TYPE ( \
        /* --- input --- */                         \
        int size,                                   \
        /* ------------- */                         \
        igo_common* igo_cm                          \
    ) ;                                             \
                                                    \
    void igo_free_vector_##TYPE (                   \
        /* --- input --- */                         \
        igo_vector_##TYPE** v_handle,               \
        /* ------------- */                         \
        igo_common* igo_cm                          \
    ) ;                                             \
                                                    \
    int igo_resize_vector_##TYPE (                  \
        /* --- input --- */                         \
        int newsize,                                \
        /* --- in/out --- */                        \
        igo_vector_##TYPE* v,                       \
        /* ------------- */                         \
        igo_common* igo_cm                          \
    ) ;                                             \
                                                    \
    int igo_vector_##TYPE##_multi_pushback (        \
        /* --- input --- */                         \
        int size,                                   \
        TYPE val,                                   \
        /* --- in/out --- */                        \
        igo_vector_##TYPE* v,                       \
        /* ------------- */                         \
        igo_common* igo_cm                          \
    ) ;                                             \

DEFINE_VECTOR_FUNCTIONS(int);
DEFINE_VECTOR_FUNCTIONS(double);

// igo_vector_int* igo_allocate_vector_int (
//     /* --- input --- */
//     int size,
//     /* ------------- */
//     igo_common* igo_cm
// ) ;
// 
// void igo_free_vector_int (
//     /* --- input --- */
//     igo_vector_int** v_handle,
//     /* ------------- */
//     igo_common* igo_cm
// ) ;
// 
// int igo_resize_vector_int (
//     /* --- input --- */
//     int newsize,
//     /* --- in/out --- */
//     igo_vector_int* v,
//     /* ------------- */
//     igo_common* igo_cm
// ) ;
// 
// int igo_vector_int_multi_pushback (
//     /* --- input --- */
//     int size,
//     int val,
//     /* --- in/out --- */
//     igo_vector_int* v,
//     /* ------------- */
//     igo_common* igo_cm
// ) ;

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
