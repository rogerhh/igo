#include <cholmod.h>
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>
#include "utils.h"

extern "C" {
#include <igo.h>
}

using namespace std;

static const double double_eps = 1e-8;

class TestCholmodUpdate2_1 : public ::testing::Test {
public:
    cholmod_common* cholmod_cm;
    cholmod_sparse* A;
    cholmod_sparse* B;
    cholmod_sparse* C;
    cholmod_sparse* D;
    cholmod_factor* LA;
    cholmod_factor* LB;

    cholmod_dense* Atb;
    cholmod_dense* delta_Atb;

    void SetUp() override {
        cholmod_cm = (cholmod_common*) malloc(sizeof(cholmod_common));
        cholmod_start(cholmod_cm);

        cholmod_cm->nmethods = 1;
        cholmod_cm->method[0].ordering = CHOLMOD_NATURAL;
        cholmod_cm->postorder = false;
        cholmod_cm->final_ll = true;

        A = cholmod_read_sparse(stdin, cholmod_cm);

        C = cholmod_read_sparse(stdin, cholmod_cm);

        D = cholmod_read_sparse(stdin, cholmod_cm);

    }

    void TearDown() override {
        cholmod_finish(cholmod_cm);
        cholmod_cm = NULL;

        cholmod_free_sparse(&A, cholmod_cm);
        A = NULL;

        cholmod_free_sparse(&B, cholmod_cm);
        B = NULL;

        cholmod_free_sparse(&C, cholmod_cm);
        C = NULL;

        cholmod_free_sparse(&D, cholmod_cm);
        D = NULL;

        cholmod_free_factor(&LA, cholmod_cm);
        LA = NULL;

        cholmod_free_factor(&LB, cholmod_cm);
        LB = NULL;

        cholmod_free_dense(&Atb, cholmod_cm);
        Atb = NULL;

        cholmod_free_dense(&delta_Atb, cholmod_cm);
        delta_Atb = NULL;
    }
};

TEST_F(TestCholmodUpdate2_1, Rank1_Updown) {
    LA = cholmod_analyze(A, cholmod_cm);
    cholmod_factorize(A, LA, cholmod_cm);

    double alpha[2] = {1, 1};
    double beta1[2] = {-1, -1};
    double beta2[2] = {1, 1};
    cholmod_sparse* B1 = cholmod_add(A, D, alpha, beta1, true, true, cholmod_cm);

    B = cholmod_add(B1, C, alpha, beta2, true, true, cholmod_cm);

    LB = cholmod_analyze(B, cholmod_cm);
    cholmod_factorize(B, LB, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LA, cholmod_cm);
    // igo_print_cholmod_factor(3, "LA before", LA, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LB, cholmod_cm);
    // igo_print_cholmod_factor(3, "LB ", LB, cholmod_cm);

    cholmod_updown2(C, D, LA, cholmod_cm);
    // igo_print_cholmod_factor(3, "LA after", LA, cholmod_cm);

    ASSERT_TRUE(igo_cholmod_factor_eq(LA, LB, double_eps, cholmod_cm));

}

TEST_F(TestCholmodUpdate2_1, Rank1_Updown_Solve) {
    LA = cholmod_analyze(A, cholmod_cm);
    cholmod_factorize(A, LA, cholmod_cm);

    double alpha[2] = {1, 1};
    double beta1[2] = {-1, -1};
    double beta2[2] = {1, 1};
    cholmod_sparse* B1 = cholmod_add(A, D, alpha, beta1, true, true, cholmod_cm);

    B = cholmod_add(B1, C, alpha, beta2, true, true, cholmod_cm);

    LB = cholmod_analyze(B, cholmod_cm);
    cholmod_factorize(B, LB, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LA, cholmod_cm);
    // igo_print_cholmod_factor(3, "LA before", LA, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LB, cholmod_cm);
    // igo_print_cholmod_factor(3, "LB ", LB, cholmod_cm);

    cholmod_updown2(C, D, LA, cholmod_cm);
    // igo_print_cholmod_factor(3, "LA after", LA, cholmod_cm);

    ASSERT_TRUE(igo_cholmod_factor_eq(LA, LB, double_eps, cholmod_cm));

}
