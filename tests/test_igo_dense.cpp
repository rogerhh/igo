#include <gtest/gtest.h>
#include <iostream>

#include "igo.h"

using namespace std;

TEST(igoDense, Contruction) {
    igo_common igo_cm;
    igo_init(&igo_cm);

    cholmod_common* cholmod_cm = igo_cm.cholmod_cm;

    int h = 6, w = 4, d = 8;

    igo_dense* igo_B = igo_allocate_dense(h, w, d, &igo_cm);

    ASSERT_NE(igo_B, nullptr);

    cholmod_dense* B = igo_B->B;

    double* Bx = (double*) B->x;

    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            ASSERT_EQ(Bx[i + j * d], 0);
        }
    }

    igo_print_dense(3, "igo_B", igo_B, &igo_cm);

    igo_free_dense(&igo_B, &igo_cm);
    ASSERT_EQ(igo_B, nullptr);

    igo_finish(&igo_cm);
}
