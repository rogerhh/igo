#include <cholmod.h>
#include <stdio.h>

void print_sparse(cholmod_sparse* A, char* name) {
    printf("Sparse matrix %s\n", name);
    // Access the data arrays
    double* values = (double*)A->x;
    int* row_indices = (int*)A->i;
    int* column_pointers = (int*)A->p;

    // Iterate through the columns
    for (int j = 0; j < A->ncol; j++) {
        int start = column_pointers[j];
        int end = column_pointers[j + 1];

        // printf("start = %d,  end = %d\n", start, end);

        // Iterate through the non-zero entries in the current column
        for (int i = start; i < end; i++) {
            
            double value = values[i];
            int row = row_indices[i];
            // printf("row = %d \n", row);

            printf("Value at (%d, %d) = %f\n", row, j, value);
        }
    }

}

int main() {
    int nrows = 3, ncols = 5, nzmax = 15;
    cholmod_common c ;
    cholmod_start(&c);
    cholmod_sparse* A = 
        cholmod_allocate_sparse(nrows, ncols, nzmax, true, true, 0, CHOLMOD_REAL, &c);

    cholmod_dense* X = NULL;
    cholmod_dense* B;

    int* Ap = (int*) A->p;
    int* Ai = (int*) A->i;
    double* Ax = (double*) A->x;

    Ap[0] = 0;
    int count = 0, next = 0;
    for(int j = 0; j < ncols; j++) {

        for(int i = 0; i < nrows; i++) {
            if(count % (nrows - 1) < 1 && next < nzmax) {
                printf("count = %d\n", count);
                Ai[next] = i;
                Ax[next] = count % (nrows + 2) + 1;
                next++;
            }
            count++;
        }
        Ap[j + 1] = next;
    }

    printf("Ap = ");
    for(int j = 0; j <= ncols; j++) {
        printf("%d ", Ap[j]);
    }
    printf("\n");

    printf("Ai = ");
    for(int i = 0; i < nzmax; i++) {
        printf("%d ", Ai[i]);
    }
    printf("\n");

    for(int i = 0; i < nzmax; i++) {
        printf("%d\n", Ai[i]);
    }

    cholmod_print_sparse(A, "A", &c);
    print_sparse(A, "A");

    cholmod_sparse* C = cholmod_aat(A, NULL, 0, 1, &c);

    cholmod_print_sparse(C, "C", &c);
    print_sparse(C, "C");

    c.supernodal = CHOLMOD_SUPERNODAL;
    cholmod_factor* L = cholmod_analyze(C, &c);
    int res = cholmod_factorize(C, L, &c);
    printf("res = %d\n", res);

    cholmod_print_factor(L, "L", &c);
    printf("%d %d %d %d %d\n", L->nsuper, L->xsize, L->ssize, L->maxcsize, L->maxesize);

    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

}
