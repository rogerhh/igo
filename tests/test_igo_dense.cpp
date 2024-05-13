#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#include "igo.h"

using namespace std;

/**
 * Tests for igo_dense functions with no pregenerated input matrices.
 * Tests include:
 * Construction - igo_init, igo_allocate_dense, igo_print_dense, igo_free_dense, igo_finish
 * Init - igo_allocate_dense, igo_allocate_dense2
 * Zeros - igo_zeros
*/
class TestIgoDense : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    cholmod_common* cholmod_cm = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);
        cholmod_cm = igo_cm->cholmod_cm;
    }

    void TearDown() override {
        igo_finish(igo_cm);
        igo_cm = nullptr;
    }
};

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

TEST_F(TestIgoDense, Init) {
    igo_dense* A = igo_allocate_dense(8, 5, 10, igo_cm);
    ASSERT_TRUE(A != NULL);
    // EXPECT_ANY_THROW({igo_dense* B = igo_allocate_dense(8, 5, 4, igo_cm);});
    cholmod_dense* chol = cholmod_allocate_dense(8, 5, 10, CHOLMOD_REAL, cholmod_cm); 
    igo_dense* C = igo_allocate_dense2(&chol, igo_cm);
    ASSERT_TRUE(C != NULL);
    ASSERT_TRUE(chol == NULL);
}

TEST_F(TestIgoDense, Zeros) {
    int nrow = 13, ncol = 15;
    igo_dense* Z = igo_zeros(nrow, ncol, CHOLMOD_REAL, igo_cm);
    double* Zarr = (double*)Z->B->x;
    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++) {
            ASSERT_TRUE(Zarr[i + j*nrow] == 0);
        }
    }
}