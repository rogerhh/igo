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

TEST_F(TestIgoDense, ResizePregenShrink) {
    cholmod_dense* cholA = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* A = igo_allocate_dense2(&cholA, igo_cm);
    cholmod_dense* cholCheck = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* check = igo_allocate_dense2(&cholCheck, igo_cm);
    igo_resize_dense(check->B->nrow, check->B->ncol, check->B->d, A, igo_cm);
    ASSERT_TRUE(igo_dense_eq(A, check, 1e-15, igo_cm));
    igo_free_dense(&A, igo_cm);
}