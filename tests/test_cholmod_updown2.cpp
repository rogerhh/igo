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
    static cholmod_common* cholmod_cm;
    static cholmod_sparse* A_orig;
    static cholmod_sparse* C_orig;
    static cholmod_sparse* D_orig;
    static cholmod_dense* b_orig;
    static cholmod_dense* b_prime_orig;
    static double alpha[2];
    static double beta1[2];
    static double beta_neg1[2];

    cholmod_sparse* A = nullptr;
    cholmod_sparse* B = nullptr;
    cholmod_sparse* C = nullptr;
    cholmod_sparse* D = nullptr;
    cholmod_factor* LA = nullptr;
    cholmod_factor* LB = nullptr;

    cholmod_dense* b = nullptr;
    cholmod_dense* b_prime = nullptr;

    static void SetUpTestSuite() {
        cholmod_cm = (cholmod_common*) malloc(sizeof(cholmod_common));
        cholmod_start(cholmod_cm);

        cholmod_cm->nmethods = 1;
        cholmod_cm->method[0].ordering = CHOLMOD_NATURAL;
        cholmod_cm->postorder = false;
        cholmod_cm->final_ll = true;

        A_orig = cholmod_read_sparse(stdin, cholmod_cm);
        C_orig = cholmod_read_sparse(stdin, cholmod_cm);
        D_orig = cholmod_read_sparse(stdin, cholmod_cm);
        b_orig = cholmod_read_dense(stdin, cholmod_cm);
        b_prime_orig = cholmod_read_dense(stdin, cholmod_cm);

    }

    static void TearDownTestSuite() {
        cholmod_free_sparse(&A_orig, cholmod_cm);
        A_orig = NULL;

        cholmod_free_sparse(&C_orig, cholmod_cm);
        C_orig = NULL;

        cholmod_free_sparse(&D_orig, cholmod_cm);
        D_orig = NULL;

        cholmod_free_dense(&b_orig, cholmod_cm);
        b_orig = NULL;

        cholmod_free_dense(&b_prime_orig, cholmod_cm);
        b_prime_orig = NULL;

        cholmod_finish(cholmod_cm);
        cholmod_cm = NULL;
    }

    void SetUp() override {
        A = cholmod_copy_sparse(A_orig, cholmod_cm);
        C = cholmod_copy_sparse(C_orig, cholmod_cm);
        D = cholmod_copy_sparse(D_orig, cholmod_cm);
        b = cholmod_copy_dense(b_orig, cholmod_cm);
        b_prime = cholmod_copy_dense(b_prime_orig, cholmod_cm);
    }

    void TearDown() override {
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

        cholmod_free_dense(&b, cholmod_cm);
        b = NULL;

        cholmod_free_dense(&b_prime, cholmod_cm);
        b_prime = NULL;
    }
};

cholmod_common* TestCholmodUpdate2_1::cholmod_cm = nullptr;
cholmod_sparse* TestCholmodUpdate2_1::A_orig = nullptr;
cholmod_sparse* TestCholmodUpdate2_1::C_orig = nullptr;
cholmod_sparse* TestCholmodUpdate2_1::D_orig = nullptr;
cholmod_dense* TestCholmodUpdate2_1::b_orig = nullptr;
cholmod_dense* TestCholmodUpdate2_1::b_prime_orig = nullptr;
double TestCholmodUpdate2_1::alpha[2] = {1, 1};
double TestCholmodUpdate2_1::beta_neg1[2] = {-1, -1};
double TestCholmodUpdate2_1::beta1[2] = {1, 1};

TEST_F(TestCholmodUpdate2_1, Updown) {
    LA = cholmod_analyze(A, cholmod_cm);
    cholmod_factorize(A, LA, cholmod_cm);

    cholmod_sparse* B1 = cholmod_add(A, D, alpha, beta_neg1, true, true, cholmod_cm);

    B = cholmod_add(B1, C, alpha, beta1, true, true, cholmod_cm);

    LB = cholmod_analyze(B, cholmod_cm);
    cholmod_factorize(B, LB, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LA, cholmod_cm);
    igo_print_cholmod_factor(3, "LA before", LA, cholmod_cm);

    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LB, cholmod_cm);
    igo_print_cholmod_factor(3, "LB ", LB, cholmod_cm);

    cholmod_updown2(C, D, LA, cholmod_cm);
    igo_print_cholmod_factor(3, "LA after", LA, cholmod_cm);

    ASSERT_TRUE(igo_cholmod_factor_eq(LA, LB, double_eps, cholmod_cm));

}

TEST_F(TestCholmodUpdate2_1, Updown2) {
    LA = cholmod_analyze(A, cholmod_cm);
    cholmod_factorize(A, LA, cholmod_cm);

    cholmod_sparse* B1 = cholmod_add(A, D, alpha, beta_neg1, true, true, cholmod_cm);

    B = cholmod_add(B1, C, alpha, beta1, true, true, cholmod_cm);

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

static cholmod_dense* cholmod_add_dense(
    cholmod_dense* B1, 
    cholmod_dense* B2, 
    cholmod_common* cm) {
    cholmod_dense* res = cholmod_allocate_dense(
                              B1->nrow, B1->ncol, B1->d, CHOLMOD_REAL, cm);
    double* B1x = (double*) B1->x;
    double* B2x = (double*) B2->x;
    double* resx = (double*) res->x;
    for(int j = 0; j < B1->ncol; j++) {
        double* B1xcol = B1x;
        double* B2xcol = B2x;
        double* resxcol = resx;
        for(int i = 0; i < B1->nrow; i++) {
            *resxcol = *B1xcol + *B2xcol;
            B1xcol++;
            B2xcol++;
            resxcol++;
        }
        B1xcol += B1->d;
        B2xcol += B2->d;
        resxcol += res->d;
    }
    return res;
}

TEST_F(TestCholmodUpdate2_1, Updown2_Solve) {
    LA = cholmod_analyze(A, cholmod_cm);
    cholmod_factorize(A, LA, cholmod_cm);
    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LA, cholmod_cm);

    // Get LALA^Tx = b
    cholmod_dense* y = cholmod_solve(4, LA, b, cholmod_cm);
    cholmod_dense* x = cholmod_solve(3, LA, y, cholmod_cm);

    // Check AA^Tx = b
    cholmod_dense* ATx = cholmod_allocate_dense(A->ncol, 1, A->ncol, CHOLMOD_REAL, cholmod_cm);
    cholmod_dense* AATx = cholmod_allocate_dense(A->nrow, 1, A->nrow, CHOLMOD_REAL, cholmod_cm);
    cholmod_sdmult(A, 1, alpha, beta1, x, ATx, cholmod_cm);
    cholmod_sdmult(A, 0, alpha, beta1, ATx, AATx, cholmod_cm);

    ASSERT_TRUE(igo_cholmod_dense_eq(AATx, b, double_eps, cholmod_cm));

    // Compute B = A + C - D and LB
    cholmod_sparse* B1 = cholmod_add(A, D, alpha, beta_neg1, true, true, cholmod_cm);

    B = cholmod_add(B1, C, alpha, beta1, true, true, cholmod_cm);

    LB = cholmod_analyze(B, cholmod_cm);
    cholmod_factorize(B, LB, cholmod_cm);
    cholmod_change_factor(CHOLMOD_REAL, false, false, true, true, LB, cholmod_cm);


    cholmod_dense* b_new = cholmod_add_dense(b, b_prime, cholmod_cm);
    cholmod_updown2_solve(C, D, LA, y, b_prime, cholmod_cm);

    cholmod_dense* y_new_cor = cholmod_solve(4, LB, b_new, cholmod_cm);

    // igo_print_cholmod_dense(3, "y_new_cor", y_new_cor, cholmod_cm);

    ASSERT_TRUE(igo_cholmod_factor_eq(LA, LB, double_eps, cholmod_cm));

    // Get LB x_new = y
    cholmod_dense* x_new = cholmod_solve(3, LA, y, cholmod_cm);
    
    // Check BB^T x_new = b_new
    cholmod_dense* ATx_new = cholmod_allocate_dense(A->ncol, 1, A->ncol, CHOLMOD_REAL, cholmod_cm);
    cholmod_dense* AATx_new = cholmod_allocate_dense(A->nrow, 1, A->nrow, CHOLMOD_REAL, cholmod_cm);
    cholmod_sdmult(B, 1, alpha, beta1, x_new, ATx_new, cholmod_cm);
    cholmod_sdmult(B, 0, alpha, beta1, ATx_new, AATx_new, cholmod_cm);

    // igo_print_cholmod_dense(3, "AATx_new", AATx_new, cholmod_cm);
    // igo_print_cholmod_dense(3, "b_new", b_new, cholmod_cm);
    ASSERT_TRUE(igo_cholmod_dense_eq(AATx_new, b_new, double_eps, cholmod_cm));

}

