#include "igo.h"

int main() {
    igo_common igo_cm;
    igo_init(&igo_cm);

    cholmod_common* cholmod_cm = igo_cm.cholmod_cm;

    cholmod_sparse* Ahat = cholmod_allocate_sparse(5, 6, 10, 1, 1, 0, CHOLMOD_REAL, cholmod_cm);

    int* Ahatp = (int*) Ahat->p;
    int* Ahati = (int*) Ahat->i;
    double* Ahatx = (double*) Ahat->x;

    Ahatp[0] = 0;
    Ahatp[1] = 1;
    Ahatp[2] = 3;
    Ahatp[3] = 5;
    Ahatp[4] = 7;
    Ahatp[5] = 7;
    Ahatp[6] = 7;

    Ahati[0] = 0;
    Ahati[1] = 0;
    Ahati[2] = 1;
    Ahati[3] = 1;
    Ahati[4] = 2;
    Ahati[5] = 2;
    Ahati[6] = 3;

    Ahatx[0] = 1;
    Ahatx[1] = 1;
    Ahatx[2] = 1;
    Ahatx[3] = 1;
    Ahatx[4] = 1;
    Ahatx[5] = 1;
    Ahatx[6] = 1;

    igo_print_cholmod_sparse(2, "Ahat", Ahat, cholmod_cm);

    cholmod_dense* Bhat = cholmod_allocate_dense(6, 1, 8, CHOLMOD_REAL, cholmod_cm);
    cholmod_dense* Atbhat = cholmod_allocate_dense(5, 1, 8, CHOLMOD_REAL, cholmod_cm);

    double* Bx = (double*) Bhat->x;

    Bx[0] = 1;
    Bx[1] = 1;
    Bx[2] = 1;
    Bx[3] = 1;
    Bx[4] = 0;
    Bx[5] = 0;

    double scale[2] = {1, 1};
    cholmod_sdmult(Ahat, 0, scale, scale, Bhat, Atbhat, cholmod_cm);

    igo_print_cholmod_dense(2, "Atb", Atbhat, cholmod_cm);

    igo_vertappend_dense(Atbhat, igo_cm.Ab, &igo_cm);
    igo_resize_dense(5, 1, 8, igo_cm.y, &igo_cm);
    igo_resize_dense(5, 1, 8, igo_cm.x, &igo_cm);

    igo_resize_factor(5, 5, igo_cm.L, &igo_cm);

    igo_print_dense(2, "orig y", igo_cm.y, &igo_cm);

    igo_sparse* igo_Ahat = igo_allocate_sparse2(&Ahat, &igo_cm);
    igo_updown_solve(1, igo_Ahat, igo_cm.L, igo_cm.y, igo_cm.Ab, &igo_cm);

    igo_print_factor(2, "updated L", igo_cm.L, &igo_cm);
    igo_print_dense(2, "updated y", igo_cm.y, &igo_cm);

    cholmod_dense* x = cholmod_solve(CHOLMOD_DLt, igo_cm.L->L, igo_cm.y->B, cholmod_cm);

    igo_print_cholmod_dense(2, "x", x, cholmod_cm);


    cholmod_free_dense(&x, cholmod_cm);
    cholmod_free_dense(&Atbhat, cholmod_cm);
    cholmod_free_dense(&Bhat, cholmod_cm);

    igo_free_sparse(&igo_Ahat, &igo_cm);

    igo_finish(&igo_cm);
}
