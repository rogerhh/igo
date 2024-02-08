#include "cholmod.h"
#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>

extern "C" {
#include "igo.h"
}

using namespace std;

static const double double_eps = 1e-8;

class TestSolveIncrement : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_Atilde = nullptr;
    igo_sparse* igo_btilde = nullptr;
    igo_sparse* igo_Ahat = nullptr;
    igo_sparse* igo_bhat = nullptr;
    igo_factor* igo_L = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        igo_Atilde = igo_allocate_sparse(0, 0, 0, igo_cm);
        igo_btilde = igo_allocate_sparse(0, 0, 0, igo_cm);

        igo_Ahat = igo_allocate_sparse(9, 9, 45, igo_cm);

        cholmod_sparse* Ahat = igo_Ahat->A;

        int* Ahatp = (int*) Ahat->p;
        int* Ahati = (int*) Ahat->i;
        double* Ahatx = (double*) Ahat->x;

        int Ahatp_setup[10] = {0, 3, 6, 9, 15, 21, 27, 33, 39, 45};
        int Ahati_setup[45] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 
                            3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 3, 4, 5, 
                            6, 7, 8, 3, 4, 5, 6, 7, 8};
        double Ahatx_setup[45] = {1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 
                               0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 
                               -0.999967, 0.008184, 0.033582, 0.000000, 1.000000, 0.000000, 
                               -0.008184, -0.999967, 0.999098, 0.000000, 0.000000, 1.000000, 
                               -0.000000, -0.000000, -1.000000, 1.000000, 0.000000, 0.000000, 
                               -0.999984, 0.005697, 0.017876, 0.000000, 1.000000, 0.000000, 
                               -0.005697, -0.999984, 1.003478, 0.000000, 0.000000, 1.000000, 
                               -0.000000, -0.000000, -1.000000};

        for(int i = 0; i < 10; i++) {
            Ahatp[i] = Ahatp_setup[i];
        }
        for(int i = 0; i < 45; i++) {
            Ahati[i] = Ahati_setup[i];
            Ahatx[i] = Ahatx_setup[i];
        }

        igo_bhat = igo_allocate_sparse(9, 1, 9, igo_cm);

        cholmod_sparse* bhat = igo_bhat->A;

        int* bhatp = (int*) bhat->p;
        int* bhati = (int*) bhat->i;
        double* bhatx = (double*) bhat->x;

        int bhatp_setup[2] = {0, 9};
        int bhati_setup[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        double bhatx_setup[9] = {0, 0, 0, -1.11022e-16, -0, 4.3644e-19, 
                              -2.22045e-16, 6.93889e-18, -1.30884e-18};

        for(int i = 0; i < 2; i++) {
            bhatp[i] = bhatp_setup[i];
        }
        for(int i = 0; i < 9; i++) {
            bhati[i] = bhati_setup[i];
            bhatx[i] = bhatx_setup[i];
        }

        igo_L = igo_allocate_factor(3, 3, igo_cm);
    }

    void TearDown() override {
        igo_free_sparse(&igo_Atilde, igo_cm);
        ASSERT_EQ(igo_Atilde, nullptr);
        
        igo_free_sparse(&igo_btilde, igo_cm);
        ASSERT_EQ(igo_btilde, nullptr);

        igo_free_sparse(&igo_Ahat, igo_cm);
        ASSERT_EQ(igo_Ahat, nullptr);

        igo_free_sparse(&igo_bhat, igo_cm);
        ASSERT_EQ(igo_Ahat, nullptr);

        igo_free_factor(&igo_L, igo_cm);
        ASSERT_EQ(igo_L, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }
};

TEST_F(TestSolveIncrement, ObsOnly) {
    igo_solve_increment2(igo_Atilde, igo_btilde, igo_Ahat, igo_bhat, igo_cm);

    double x_cor[9] = {-4.44086e-28, 5.11888e-30, -8.75298e-30,
                       1.11e-16, -1.34449e-18,  -4.3644e-19,
                       3.33013e-16, -9.3052e-18, 8.72396e-19};

    double* xx = (double*) igo_cm->x->B->x;

    ASSERT_EQ(igo_cm->x->B->nrow, 9);
    ASSERT_EQ(igo_cm->x->B->ncol, 1);

    for(int i = 0; i < 9; i++) {
        EXPECT_NEAR(xx[i], x_cor[i], 1e-8);
    }

}

class TestSolveIncrement_FromFile : public ::testing::Test {
public:
    static igo_common* igo_cm;
    static igo_sparse* A_orig;
    static igo_sparse* b_orig;
    static igo_sparse* A_tilde_orig;
    static igo_sparse* b_tilde_orig;
    static igo_sparse* A_hat_orig;
    static igo_sparse* b_hat_orig;
    static double alpha[2];
    static double beta1[2];
    static double beta_neg1[2];

    igo_sparse* A = NULL;
    igo_sparse* b = NULL;
    igo_sparse* A_tilde = NULL;
    igo_sparse* b_tilde = NULL;
    igo_sparse* A_hat = NULL;
    igo_sparse* b_hat = NULL;

    static void SetUpTestSuite() {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        cholmod_sparse* A = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, A, igo_cm->cholmod_cm);
        A_orig = igo_allocate_sparse2(&A, igo_cm);
        cholmod_sparse* b = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, b, igo_cm->cholmod_cm);
        b_orig = igo_allocate_sparse2(&b, igo_cm);
        cholmod_sparse* A_tilde = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, A_tilde, igo_cm->cholmod_cm);
        A_tilde_orig = igo_allocate_sparse2(&A_tilde, igo_cm);
        cholmod_sparse* b_tilde = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, b_tilde, igo_cm->cholmod_cm);
        b_tilde_orig = igo_allocate_sparse2(&b_tilde, igo_cm);
        cholmod_sparse* A_hat = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, A_hat, igo_cm->cholmod_cm);
        A_hat_orig = igo_allocate_sparse2(&A_hat, igo_cm);
        cholmod_sparse* b_hat = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, b_hat, igo_cm->cholmod_cm);
        b_hat_orig = igo_allocate_sparse2(&b_hat, igo_cm);
    }

    static void TearDownTestSuite() {
        igo_free_sparse(&A_orig, igo_cm);
        A_orig = NULL;
        igo_free_sparse(&b_orig, igo_cm);
        b_orig = NULL;
        igo_free_sparse(&A_tilde_orig, igo_cm);
        A_tilde_orig = NULL;
        igo_free_sparse(&b_tilde_orig, igo_cm);
        b_tilde_orig = NULL;
        igo_free_sparse(&A_hat_orig, igo_cm);
        A_hat_orig = NULL;
        igo_free_sparse(&b_hat_orig, igo_cm);
        b_hat_orig = NULL;
        igo_finish(igo_cm);
        igo_cm = NULL;
    }

    void SetUp() override {
        A = igo_copy_sparse(A_orig, igo_cm);
        b = igo_copy_sparse(b_orig, igo_cm);
        A_tilde = igo_copy_sparse(A_tilde_orig, igo_cm);
        b_tilde = igo_copy_sparse(b_tilde_orig, igo_cm);
        A_hat = igo_copy_sparse(A_hat_orig, igo_cm);
        b_hat = igo_copy_sparse(b_hat_orig, igo_cm);
    }

    void TearDown() override {
        igo_free_sparse(&A, igo_cm);
        A = NULL;
        igo_free_sparse(&b, igo_cm);
        b = NULL;
        igo_free_sparse(&A_tilde, igo_cm);
        A_tilde = NULL;
        igo_free_sparse(&b_tilde, igo_cm);
        b_tilde = NULL;
        igo_free_sparse(&A_hat, igo_cm);
        A_hat = NULL;
        igo_free_sparse(&b_hat, igo_cm);
        b_hat = NULL;
    }
};

igo_common* TestSolveIncrement_FromFile::igo_cm = NULL;
igo_sparse* TestSolveIncrement_FromFile::A_orig = NULL;
igo_sparse* TestSolveIncrement_FromFile::b_orig = NULL;
igo_sparse* TestSolveIncrement_FromFile::A_tilde_orig = NULL;
igo_sparse* TestSolveIncrement_FromFile::b_tilde_orig = NULL;
igo_sparse* TestSolveIncrement_FromFile::A_hat_orig = NULL;
igo_sparse* TestSolveIncrement_FromFile::b_hat_orig;
double TestSolveIncrement_FromFile::alpha[2] = {1, 1};
double TestSolveIncrement_FromFile::beta1[2] = {1, 1};
double TestSolveIncrement_FromFile::beta_neg1[2] = {-1, -1};

TEST_F(TestSolveIncrement_FromFile, ObsAndUpdate) {
    // igo_print_sparse(3, "A", A, igo_cm);

    igo_sparse* A_null = igo_allocate_sparse(0, 0, 0, igo_cm);
    igo_sparse* b_null = igo_allocate_sparse(0, 1, 0, igo_cm);

    igo_solve_increment2(A_null, b_null, A, b, igo_cm);

    int h = A->A->nrow;
    int w = A->A->ncol;
    int h_new = A_hat->A->nrow;
    int w_new = w + A_hat->A->ncol;

    {

        igo_dense* ATx = igo_allocate_dense(w, 1, w, igo_cm);
        igo_dense* AATx = igo_allocate_dense(h, 1, h, igo_cm);
        igo_dense* Ab = igo_allocate_dense(h, 1, h, igo_cm);

        double beta1[2] = {1, 1};
        double beta_neg1[2] = {-1, -1};

        igo_sdmult(A, 1, beta1, beta1, igo_cm->x, ATx, igo_cm);
        igo_sdmult(A, 0, beta1, beta1, ATx, AATx, igo_cm);
        igo_sdmult(A, 0, beta1, beta1, igo_cm->b, Ab, igo_cm);

        igo_print_dense(3, "AATx", AATx, igo_cm);
        // igo_print_dense(3, "b", igo_cm->b, igo_cm);
        igo_print_dense(3, "Ab", Ab, igo_cm);

        ASSERT_TRUE(igo_dense_eq(AATx, Ab, double_eps, igo_cm));

    }

    igo_solve_increment2(A_tilde, b_tilde, A_hat, b_hat, igo_cm);

    {
        igo_sparse* A_copy = igo_copy_sparse(A_orig, igo_cm);
        igo_sparse* A_tilde_neg = igo_replace_sparse(A_copy, A_tilde, igo_cm);
        igo_horzappend_sparse2(A_hat, A_copy, igo_cm);

        cholmod_dense* cholmod_b_copy = cholmod_sparse_to_dense(b_orig->A, igo_cm->cholmod_cm);
        igo_dense* b_copy = igo_allocate_dense2(&cholmod_b_copy, igo_cm);;

        int b_tilde_nz = ((int*) b_tilde->A->p)[1];
        int* b_tilde_i = (int*) b_tilde->A->i;
        double* b_tilde_x = (double*) b_tilde->A->x;
        double* b_copy_x = (double*) b_copy->B->x;

        for(int i = 0; i < b_tilde_nz; i++) {
            int brow = b_tilde_i[i];
            b_copy_x[brow] = b_tilde_x[i];
        }

        cholmod_dense* cholmod_dense_b_hat = cholmod_sparse_to_dense(b_hat->A, igo_cm->cholmod_cm);
        igo_dense* dense_b_hat = igo_allocate_dense2(&cholmod_dense_b_hat, igo_cm);
        igo_vertappend_dense2(dense_b_hat, b_copy, igo_cm);
        // igo_resize_dense(w_new, 1, w_new, b_copy, igo_cm);
        // int b_hat_nz = ((int*) b_hat->A->p)[1];
        // int* b_hat_i = (int*) b_hat->A->i;
        // double* b_hat_x = (double*) b_hat->A->x;
        // b_copy_x = (double*) b_copy->B->x;

        // for(int i = 0; i < b_hat_nz; i++) {
        //     int brow = b_hat_i[i];
        //     b_copy_x[brow] = b_hat_x[i];
        // }

        igo_sparse* PA = igo_submatrix(A_copy, (int*) (igo_cm->L->L->Perm), A_copy->A->nrow, NULL, -1, true, true, igo_cm);
        
        // igo_cm->cholmod_cm->nmethods = 1;
        // igo_cm->cholmod_cm->method[0].ordering = CHOLMOD_NATURAL;
        // igo_cm->cholmod_cm->postorder = false;

        // igo_factor* LPA = igo_analyze_and_factorize(PA, igo_cm);

        // igo_print_factor(3, "L", igo_cm->L, igo_cm);
        // igo_print_factor(3, "LPA", LPA, igo_cm);
        // assert(igo_factor_eq(igo_cm->L, LPA, double_eps, igo_cm));

        igo_dense* ATx = igo_allocate_dense(w_new, 1, w_new, igo_cm);
        igo_dense* AATx = igo_allocate_dense(h_new, 1, h_new, igo_cm);
        igo_dense* Ab = igo_allocate_dense(h_new, 1, h_new, igo_cm);

        igo_sdmult(A_copy, 1, beta1, beta1, igo_cm->x, ATx, igo_cm);
        igo_sdmult(A_copy, 0, beta1, beta1, ATx, AATx, igo_cm);
        igo_sdmult(A_copy, 0, beta1, beta1, b_copy, Ab, igo_cm);

        // igo_print_dense(3, "AATx", AATx, igo_cm);
        // igo_print_dense(3, "b", igo_cm->b, igo_cm);
        // igo_print_dense(3, "Ab", Ab, igo_cm);

        ASSERT_TRUE(igo_dense_eq(AATx, Ab, double_eps, igo_cm));
    
    }

}
