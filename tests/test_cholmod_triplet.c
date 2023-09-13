#include "igo.h"

int main() {
    cholmod_common cm;
    cholmod_start(&cm);

    cholmod_triplet* t = cholmod_allocate_triplet(5, 5, 10, 0, CHOLMOD_REAL, &cm);

    int* ti = (int*) t->i;
    int* tj = (int*) t->j;
    double* tx = (double*) t->x;

    printf("%d %d\n", sizeof(int), sizeof(size_t));
    memset(ti, 0, t->nzmax * sizeof(int));
    memset(tj, 0, t->nzmax * sizeof(int));
    memset(tx, 0, t->nzmax * sizeof(double));

    t->nnz = 0;

    ti[t->nnz] = 3;
    tj[t->nnz] = 2;
    tx[t->nnz] = 132;
    t->nnz++;

    ti[1] = 0;
    tj[1] = 0;
    tx[1] = 0;

    ti[2] = 2;
    tj[2] = 3;
    tx[2] = 100;
    t->nnz = 3;

    cholmod_print_triplet(t, "t", &cm);

    for(int i = 0; i < t->nzmax; i++) {
        printf("%d %d %f\n", ti[i], tj[i], tx[i]);
    }


    cholmod_sparse* a = cholmod_triplet_to_sparse(t, t->nzmax, &cm);

    igo_print_cholmod_sparse(3, "a", a, &cm);

    cholmod_free_triplet(&t, &cm);
    cholmod_free_sparse(&a, &cm);
    

    cholmod_finish(&cm);
}
