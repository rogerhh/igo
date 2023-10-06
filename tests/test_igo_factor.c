#include "igo.h"

int main() {
    igo_common igo_cm;
    igo_init(&igo_cm);

    cholmod_common* cholmod_cm = igo_cm.cholmod_cm;

    igo_sparse* igo_A = igo_allocate_sparse(5, 6, 10, &igo_cm);

    cholmod_sparse* A = igo_A->A;

    int* Ap = (int*) A->p;
    int* Ai = (int*) A->i;
    double* Ax = (double*) A->x;

    Ap[0] = 0;
    Ap[1] = 1;
    Ap[2] = 3;
    Ap[3] = 5;
    Ap[4] = 7;
    Ap[5] = 7;
    Ap[6] = 7;

    Ai[0] = 0;
    Ai[1] = 0;
    Ai[2] = 1;
    Ai[3] = 1;
    Ai[4] = 2;
    Ai[5] = 2;
    Ai[6] = 3;

    Ax[0] = 1;
    Ax[1] = 1;
    Ax[2] = 1;
    Ax[3] = 1;
    Ax[4] = 1;
    Ax[5] = 1;
    Ax[6] = 1;

    igo_print_sparse(1, "A", igo_A, &igo_cm);

    igo_factor* igo_L = igo_allocate_factor(3, 3, &igo_cm);

    igo_print_factor(2, "L", igo_L, &igo_cm);

    igo_updown(1, igo_A, igo_L, &igo_cm);

    cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, igo_L->L, cholmod_cm);

    igo_print_factor(3, "L", igo_L, &igo_cm);

    igo_free_factor(&igo_L, &igo_cm);

    igo_free_sparse(&igo_A, &igo_cm);

    igo_finish(&igo_cm);
}
