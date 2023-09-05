#include "igo.h"

int main() {
    igo_common igo_cm;
    igo_init(&igo_cm);

    igo_dense* igo_B = igo_allocate_dense(5, 4, 8, &igo_cm);

    cholmod_common* cholmod_cm = igo_cm.cholmod_cm;

    cholmod_dense* B = igo_B->B;

    double* Bx = (double*) B->x;
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 4; j++) {
            Bx[j * B->d + i] = i * 10 + j;
        }
    }

    igo_print_dense(2, "B", igo_B, &igo_cm);

    cholmod_dense* Bhat = cholmod_allocate_dense(4, 3, 6, CHOLMOD_REAL, cholmod_cm);

    double* Bhatx = (double*) Bhat->x;
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 3; j++) {
            Bhatx[j * Bhat->d + i] = i * 10 + j;
        }
    }

    igo_print_cholmod_dense(2, "Bhat", Bhat, cholmod_cm);

    igo_vertappend_dense(Bhat, igo_B, &igo_cm);

    igo_print_dense(2, "B", igo_B, &igo_cm);

    cholmod_free_dense(&Bhat, cholmod_cm);
    igo_free_dense(&igo_B, &igo_cm);

    igo_finish(&igo_cm);
}
