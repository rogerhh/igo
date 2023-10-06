#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>

extern "C" {
#include "igo.h"
}

using namespace std;

class TestIgoSparse1 : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_A = nullptr;
    igo_sparse* igo_B = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        igo_A = igo_allocate_sparse(5, 6, 10, igo_cm);

        cholmod_sparse* A = igo_A->A;

        int* Ap = (int*) A->p;
        int* Ai = (int*) A->i;
        double* Ax = (double*) A->x;

        Ap[0] = 0;
        Ap[1] = 1;
        Ap[2] = 3;
        Ap[3] = 5;
        Ap[4] = 7;
        Ap[5] = 7;
        Ap[6] = 7;

        Ai[0] = 0;
        Ai[1] = 0;
        Ai[2] = 1;
        Ai[3] = 1;
        Ai[4] = 2;
        Ai[5] = 2;
        Ai[6] = 3;

        Ax[0] = 1;
        Ax[1] = 1;
        Ax[2] = 1;
        Ax[3] = 1;
        Ax[4] = 1;
        Ax[5] = 1;
        Ax[6] = 1;

        igo_B = igo_allocate_sparse(5, 6, 10, igo_cm);

        cholmod_sparse* B = igo_B->A;

        int* Bp = (int*) B->p;
        int* Bi = (int*) B->i;
        double* Bx = (double*) B->x;

        Bp[0] = 0;
        Bp[1] = 1;
        Bp[2] = 3;
        Bp[3] = 5;
        Bp[4] = 7;
        Bp[5] = 7;
        Bp[6] = 7;

        Bi[0] = 0;
        Bi[1] = 0;
        Bi[2] = 1;
        Bi[3] = 1;
        Bi[4] = 2;
        Bi[5] = 2;
        Bi[6] = 3;

        Bx[0] = 2;
        Bx[1] = 2;
        Bx[2] = 2;
        Bx[3] = 2;
        Bx[4] = 2;
        Bx[5] = 2;
        Bx[6] = 2;

    }

    void TearDown() override {
        igo_free_sparse(&igo_A, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_sparse(&igo_B, igo_cm);
        ASSERT_EQ(igo_B, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }

};

TEST_F(TestIgoSparse1, VertAppendSparse) {


    
    igo_print_sparse(3, "A Before", igo_A, igo_cm);
    igo_vertappend_sparse(igo_B->A, igo_A, igo_cm);
    igo_print_sparse(3, "A After", igo_A, igo_cm);

    int nrow_cor = 10, ncol_cor = 6, nzmax_cor = 20;
    int Ap_cor[7] = {0, 2, 6, 10, 14, 14, 14}; 
    int Ai_cor[14] = {0, 5, 0, 1, 5, 6, 1, 2, 6, 7, 2, 3, 7, 8}; 
    double Ax_cor[14] = {1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2}; 

    int* Ap = (int*) igo_A->A->p;
    int* Ai = (int*) igo_A->A->i;
    double* Ax = (double*) igo_A->A->x;
    for(int i = 0; i < 7; i++) {
        ASSERT_EQ(Ap[i], Ap_cor[i]);
    }
    for(int i = 0; i < 14; i++) {
        ASSERT_EQ(Ai[i], Ai_cor[i]);
    }
    for(int i = 0; i < 14; i++) {
        ASSERT_EQ(Ax[i], Ax_cor[i]);
    }

}

TEST_F(TestIgoSparse1, VertAppendSparse2) {

    int res = 0;

    igo_print_sparse(3, "A Before", igo_A, igo_cm);
    res = igo_vertappend_sparse2(igo_B, igo_A, igo_cm);
    ASSERT_EQ(res, 1);
    igo_print_sparse(3, "A After", igo_A, igo_cm);

    int nrow_cor = 10, ncol_cor = 6, nzmax_cor = 20;
    int Ap_cor[7] = {0, 2, 6, 10, 14, 14, 14}; 
    int Ai_cor[14] = {0, 5, 0, 1, 5, 6, 1, 2, 6, 7, 2, 3, 7, 8}; 
    double Ax_cor[14] = {1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2}; 

    int* Ap = (int*) igo_A->A->p;
    int* Ai = (int*) igo_A->A->i;
    double* Ax = (double*) igo_A->A->x;
    for(int i = 0; i < 7; i++) {
        ASSERT_EQ(Ap[i], Ap_cor[i]);
    }
    for(int i = 0; i < 14; i++) {
        ASSERT_EQ(Ai[i], Ai_cor[i]);
    }
    for(int i = 0; i < 14; i++) {
        ASSERT_EQ(Ax[i], Ax_cor[i]);
    }

}

TEST_F(TestIgoSparse1, ResizeSmaller) {
    int res = 0;
    igo_print_sparse(3, "A Before", igo_A, igo_cm);
    res = igo_resize_sparse(5, 2, 10, igo_A, igo_cm);
    ASSERT_EQ(res, 1);
    igo_print_sparse(3, "A After", igo_A, igo_cm);

    int nrow_cor = 5, ncol_cor = 2, nzmax_cor = 10;
    int Ap_cor[3] = {0, 1, 3}; 
    int Ai_cor[3] = {0, 0, 1};
    double Ax_cor[3] = {1, 1, 1};

    int* Ap = (int*) igo_A->A->p;
    int* Ai = (int*) igo_A->A->i;
    double* Ax = (double*) igo_A->A->x;
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(Ap[i], Ap_cor[i]);
    }
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(Ai[i], Ai_cor[i]);
    }
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(Ax[i], Ax_cor[i]);
    }
}
