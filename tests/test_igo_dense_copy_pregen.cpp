#include <gtest/gtest.h>
#include <iostream>
#include <fstream>

#include "igo.h"

using namespace std;

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

TEST_F(TestIgoDense, CopyRand) {
    cholmod_dense* chol = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* A = igo_allocate_dense2(&chol, igo_cm);
    igo_dense* B = igo_copy_dense(A, igo_cm);
    ASSERT_TRUE(igo_dense_eq(A, B, 1e-15, igo_cm));
    igo_free_dense(&A, igo_cm);
    igo_free_dense(&B, igo_cm);
}