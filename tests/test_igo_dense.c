#include "igo.h"

int main() {
    igo_common igo_cm;
    igo_init(&igo_cm);

    cholmod_common* cholmod_cm = igo_cm.cholmod_cm;

    igo_dense* igo_B = igo_allocate_dense(6, 1, 6, &igo_cm);

    igo_print_dense(2, "igo_B", igo_B, &igo_cm);

    igo_free_dense(&igo_B, &igo_cm);

    igo_finish(&igo_cm);
}
