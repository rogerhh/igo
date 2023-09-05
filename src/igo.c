#include "igo.h"

int igo_init (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_cm->cholmod_cm = malloc(sizeof(cholmod_common));
    cholmod_start(igo_cm->cholmod_cm);

    igo_cm->A = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_cm->delta_Atb = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->L = igo_allocate_factor(0, 0, igo_cm);
    igo_cm->x = igo_allocate_dense(0, 0, 0, igo_cm);
    igo_cm->y = igo_allocate_dense(0, 0, 0, igo_cm);

    return 1;
}

int igo_finish (
    /* --- inouts --- */
    igo_common* igo_cm
) {
    igo_free_sparse(&(igo_cm->A), igo_cm);
    igo_free_dense(&(igo_cm->delta_Atb), igo_cm);
    igo_free_factor(&(igo_cm->L), igo_cm);
    igo_free_dense(&(igo_cm->x), igo_cm);
    igo_free_dense(&(igo_cm->y), igo_cm);

    cholmod_finish(igo_cm->cholmod_cm);
    free(igo_cm->cholmod_cm);
    igo_cm->cholmod_cm = NULL;
    return 1;
}

int igo_solve_increment (
    /* --- inputs --- */   
    igo_sparse* A_tilde, 
    igo_dense* b_tilde,
    igo_sparse* A_hat,
    igo_dense* b_hat,
    /* --- outputs --- */
    cholmod_dense* x,
    /* --- common --- */
    igo_common* igo_cm
) {
    if(igo_cm == NULL) {
        return 0;
    }

    igo_horzappend_sparse(A_tilde, A_hat, igo_cm);
    igo_vertappend_sparse(b_tilde, b_hat, igo_cm);

    // Do an update on Ab_hat first
    // igo_updown_solve(1, A_hat, igo_cm->L, x, b_hat, igo_cm->cholmod_cm);

    // Then do an update on Ab_tilde
    // igo_updown_solve(-1, A_tilde, igo_cm->L, x, b_tilde, igo_cm->cholmod_cm);

    return 1;
}

void igo_print_cholmod_sparse(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_sparse* A,
    cholmod_common* cholmod_cm
) {
    cholmod_print_sparse(A, name, cholmod_cm);

    if(!verbose) 
        return; 

    if(verbose >= 2) {

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
        for(int i = 0; i < A->nzmax; i++) {
            printf("%d ", Ai[i]);
        }
        printf("\n");
    }

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

            printf("Value at (%d, %d) = %f\n", row, j, value);
        }
    }
}

/* 
Returns A = [A B]. This is needed as cholmod_horzcat copies the input matrices
*/
int igo_horzappend (
    /* --- input --- */  
    cholmod_sparse* B,
    /* --- in/out --- */
    cholmod_sparse* A
) {

}

/*
Trivially add rows to a sparse matrix
*/
int igo_sparse_addrows(
    /* --- input --- */
    int nrow,       /* Number of total rows to add */
    /* --- in/out --- */
    cholmod_sparse* A
) {
    A->nrow = nrow;
    return 1;
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
                printf("%f ", Bx[j * B->d + i]);
            }
            printf("\n");
        }
    }
}


void igo_print_cholmod_factor(
    /* --- input --- */
    int verbose,
    char* name,
    cholmod_factor* L,
    cholmod_common* cholmod_cm
) {
    cholmod_print_factor(L, name, cholmod_cm);

    bool is_ll_old = L->is_ll;

    cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, cholmod_cm);

    if(verbose >= 3) {
        printf("itype = %d, xtype = %d, dtype = %d\n", L->itype, L->xtype, L->dtype);
        printf("ordering = %d, is_ll = %d, is_super = %d, is_monotonic = %d\n", L->ordering, L->is_ll, L->is_super, L->is_monotonic);
    }

    if(verbose >= 2) {
        int* Lp = (int*) L->p;
        int* Li = (int*) L->i;
        double* Lx = (double*) L->x;
        int* LPerm = (int*) L->Perm;
        int* Lnext = (int*) L->next;
        int* Lprev = (int*) L->prev;

        printf("Lp = ");
        for(int j = 0; j < L->n + 1; j++) {
            printf("%d ", Lp[j]);
        }
        printf("\n");
        printf("Li = ");
        for(int j = 0; j < L->nzmax; j++) {
            printf("%d ", Li[j]);
        }
        printf("\n");
        printf("Lx = ");
        for(int j = 0; j < L->nzmax; j++) {
            printf("%f ", Lx[j]);
        }
        printf("\n");
        printf("Next = ");
        for(int j = 0; j < L->n + 2; j++) {
            printf("%d ", Lnext[j]);
        }
        printf("\n");
        printf("Prev = ");
        for(int j = 0; j < L->n + 2; j++) {
            printf("%d ", Lprev[j]);
        }
        printf("\n");
        printf("Perm = ");
        for(int j = 0; j < L->n; j++) {
            printf("%d ", LPerm[j]);
        }
        printf("\n");
    }

    if(verbose >= 1) {
        if(!L->is_super) {
            // Access the data arrays
            double* values = (double*)L->x;
            int* row_indices = (int*)L->i;
            int* column_pointers = (int*)L->p;

            // Iterate through the columns
            for (int j = 0; j < L->n; j++) {
                int start = column_pointers[j];
                int end = column_pointers[j + 1];

                // Iterate through the non-zero entries in the current column
                for (int i = start; i < end; i++) {

                    double value = values[i];
                    int row = row_indices[i];
                    // printf("row = %d \n", row);

                    printf("Value at (%d, %d) = %f\n", row, j, value);
                }
            }
        }
        else {
            int* Lsuper = (int*) L->super;
            int* Lpi = (int*) L->pi;
            int* Lpx = (int*) L->px;
            int* Ls = (int*) L->s;
            double* Lx = (double*) L->x;
            for(int js = 0; js < L->nsuper; js++) {
                int super_width = Lsuper[js + 1] - Lsuper[js];
                printf("Supernode: ");
                for(int j = 0; j < super_width; j++) {
                    printf("%d ", Lsuper[js] + j);
                }
                printf("\n");

                int super_height = Lpi[js + 1] - Lpi[js];
                for(int i = 0; i < super_height; i++) {
                    printf("Row %d: ", Ls[Lpi[js] + i]);

                    for(int j = 0; j < super_width; j++) {
                        int idx = Lpx[js] + i + j * super_height;
                        printf("%f ", Lx[idx]);
                    }

                    printf("\n");
                }
            }
        }
    }

    cholmod_change_factor(CHOLMOD_REAL, is_ll_old, 0, 1, 1, L, cholmod_cm);
}

