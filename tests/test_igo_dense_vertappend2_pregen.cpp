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

TEST_F(TestIgoDense, VertAppend2Pregen) {
    cholmod_dense* cholA = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* A = igo_allocate_dense2(&cholA, igo_cm);
    cholmod_dense* cholB = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* B = igo_allocate_dense2(&cholB, igo_cm);
    cholmod_dense* cholC = cholmod_read_dense(stdin, cholmod_cm);
    igo_dense* C = igo_allocate_dense2(&cholC, igo_cm);
    igo_vertappend_dense2(B, A, igo_cm);
    ASSERT_TRUE(igo_dense_eq(A, C, 1e-15, igo_cm));
    igo_free_dense(&A, igo_cm);
    igo_free_dense(&B, igo_cm);
    igo_free_dense(&C, igo_cm);
}