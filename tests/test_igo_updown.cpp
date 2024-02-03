#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>

extern "C" {
#include "igo.h"
}

class TestIgoUpdown : public ::testing::Test {
public:
    
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_A = nullptr;
    igo_sparse* igo_b = nullptr;
    igo_dense* igo_Ab = nullptr;
    igo_dense* igo_y = nullptr;
    igo_factor* igo_L = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        igo_A = igo_allocate_sparse(2, 3, 6, igo_cm);

        cholmod_sparse* A = igo_A->A;

        int* Ap = (int*) A->p;
        int* Ai = (int*) A->i;
        double* Ax = (double*) A->x;

        Ap[0] = 0;
        Ap[1] = 2;
        Ap[2] = 4;
        Ap[3] = 6;

        Ai[0] = 0;
        Ai[1] = 1;
        Ai[2] = 0;
        Ai[3] = 1;
        Ai[4] = 0;
        Ai[5] = 1;

        Ax[0] = 1;
        Ax[1] = 1;
        Ax[2] = 2;
        Ax[3] = 3;
        Ax[4] = 0;
        Ax[5] = 0;

        igo_b = igo_allocate_sparse(3, 1, 3, igo_cm);

        cholmod_sparse* b = igo_b->A;

        int* bp = (int*) b->p;
        int* bi = (int*) b->i;
        double* bx = (double*) b->x;

        bp[0] = 0;
        bp[1] = 3;

        bi[0] = 0;
        bi[1] = 1;
        bi[2] = 2;

        bx[0] = 1;
        bx[1] = 1;
        bx[2] = 0;

        igo_Ab = igo_allocate_dense(2, 1, 2, igo_cm);
        igo_y = igo_allocate_dense(2, 1, 2, igo_cm);
        igo_L = igo_allocate_factor(2, 2, igo_cm);

    }

    void TearDown() override {
        igo_free_sparse(&igo_A, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_dense(&igo_Ab, igo_cm);
        ASSERT_EQ(igo_Ab, nullptr);

        igo_free_dense(&igo_y, igo_cm);
        ASSERT_EQ(igo_y, nullptr);

        igo_free_factor(&igo_L, igo_cm);
        ASSERT_EQ(igo_L, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }

};

TEST_F(TestIgoUpdown, update1) {
    igo_sparse* igo_sparse_Ab = igo_ssmult(igo_A, igo_b, 0, true, true, igo_cm); 

    int* Abp = (int*) igo_sparse_Ab->A->p;
    int* Abi = (int*) igo_sparse_Ab->A->i;
    double* Abx = (double*) igo_sparse_Ab->A->x;
    double* dense_Abx = (double*) igo_Ab->B->x;
    for(int j = 0; j < 1; j++) {
        for(int idx = Abp[0]; idx < Abp[1]; idx++) {
            int i = Abi[idx];
            int x = Abx[idx];
            dense_Abx[i] = x;
        }
    }

    igo_print_dense(3, "igo_Ab", igo_Ab, igo_cm);

    igo_updown_solve(1, igo_A, igo_L, igo_y, igo_Ab, igo_cm);

    igo_print_factor(3, "igo_L after updown solve 1", igo_L, igo_cm);

    igo_print_dense(3, "igo_y", igo_y, igo_cm);

    igo_dense* igo_x = igo_solve(CHOLMOD_DLt, igo_L, igo_y, igo_cm);
    igo_print_dense(3, "igo_x", igo_x, igo_cm);

    double* xx = (double*) igo_x->B->x;

    ASSERT_FLOAT_EQ(xx[0], 2);
    ASSERT_FLOAT_EQ(xx[1], -1);

    igo_free_dense(&igo_x, igo_cm);

    igo_sparse* igo_A_tilde = igo_allocate_sparse(2, 2, 6, igo_cm);

    int* Atilde_p = (int*) igo_A_tilde->A->p;
    int* Atilde_i = (int*) igo_A_tilde->A->i;
    double* Atilde_x = (double*) igo_A_tilde->A->x;

    Atilde_p[0] = 0;
    Atilde_p[1] = 2;
    Atilde_p[2] = 4;

    Atilde_i[0] = 0;
    Atilde_i[1] = 1;
    Atilde_i[2] = 0;
    Atilde_i[3] = 1;

    Atilde_x[0] = 0;
    Atilde_x[1] = 0;
    Atilde_x[2] = 3;
    Atilde_x[3] = 2;

    dense_Abx[0] = 3;
    dense_Abx[1] = 2;

    igo_updown_solve(1, igo_A_tilde, igo_L, igo_y, igo_Ab, igo_cm);

    igo_x = igo_solve(CHOLMOD_DLt, igo_L, igo_y, igo_cm);
    igo_print_dense(3, "igo_x", igo_x, igo_cm);

    igo_free_dense(&igo_x, igo_cm);

    Atilde_x[0] = 0;
    Atilde_x[1] = 0;
    Atilde_x[2] = 2;
    Atilde_x[3] = 3;

    dense_Abx[0] = -2;
    dense_Abx[1] = -3;

    igo_updown_solve(false, igo_A_tilde, igo_L, igo_y, igo_Ab, igo_cm);

    igo_print_factor(3, "igo_L after updown solve", igo_L, igo_cm);

    igo_x = igo_solve(CHOLMOD_DLt, igo_L, igo_y, igo_cm);
    igo_print_dense(3, "igo_x", igo_x, igo_cm);

    igo_free_dense(&igo_x, igo_cm);

    igo_free_sparse(&igo_sparse_Ab, igo_cm);
    ASSERT_EQ(igo_sparse_Ab, nullptr);

    igo_free_sparse(&igo_A_tilde, igo_cm);
    ASSERT_EQ(igo_A_tilde, nullptr);

}
