#include "cholmod.h"
#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>
#include <cmath>

extern "C" {
#include "igo.h"
}

using namespace std;

class TestIgoFactor1 : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_A = nullptr;
    igo_factor* igo_L = nullptr;

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

        igo_L = igo_allocate_factor(3, 3, igo_cm);
    }

    void TearDown() override {
        igo_free_sparse(&igo_A, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_factor(&igo_L, igo_cm);
        ASSERT_EQ(igo_L, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }
};

TEST_F(TestIgoFactor1, Init) {
    int Lp_cor[4] = {0, 1, 2, 3};
    int Li_cor[3] = {0, 1, 2};
    double Lx_cor[3] = {1e-12, 1e-12, 1e-12};
    int Next_cor[5] = {1, 2, 3, -1, 0};
    int Prev_cor[5] = {4, 0, 1, 2, -1};
    int Perm_cor[3] = {0, 1, 2};

    cholmod_factor* L = igo_L->L;

    int* Lp = (int*) L->p;
    int* Li = (int*) L->i;
    double* Lx = (double*) L->x;
    int* LPerm = (int*) L->Perm;
    int* Lnext = (int*) L->next;
    int* Lprev = (int*) L->prev;

    ASSERT_EQ(L->n, 3);

    for(int i = 0; i < 4; i++) {
        ASSERT_EQ(Lp[i], Lp_cor[i]);
    }
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(Li[i], Li_cor[i]);
    }
    for(int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(Lx[i], Lx_cor[i]);
    }
    for(int i = 0; i < 5; i++) {
        ASSERT_EQ(Lnext[i], Next_cor[i]);
    }
    for(int i = 0; i < 5; i++) {
        ASSERT_EQ(Lprev[i], Prev_cor[i]);
    }
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(LPerm[i], Perm_cor[i]);
    }
}

TEST_F(TestIgoFactor1, Updown) {
    igo_updown(1, igo_A, igo_L, igo_cm);

    igo_print_factor(3, "igo_L", igo_L, igo_cm);

    int Lp_cor[6] = {0, 2, 4, 6, 7, 8};
    int Li_cor[8] = {0, 1, 1, 2, 2, 3, 3, 4};
    double Lx_cor[8] = {2, 0.5, 1.5, 1 / 1.5, 2 / 1.5, 1.5/ 2, 0.25, 1e-12};
    int Next_cor[7] = {1, 2, 3, 4, 5, -1, 0};
    int Prev_cor[7] = {6, 0, 1, 2, 3, 4, -1};
    int Perm_cor[5] = {0, 1, 2, 3, 4};

    cholmod_factor* L = igo_L->L;

    int* Lp = (int*) L->p;
    int* Li = (int*) L->i;
    double* Lx = (double*) L->x;
    int* LPerm = (int*) L->Perm;
    int* Lnext = (int*) L->next;
    int* Lprev = (int*) L->prev;

    ASSERT_EQ(L->n, 5);

    for(int i = 0; i < 6; i++) {
        ASSERT_EQ(Lp[i], Lp_cor[i]);
    }
    for(int i = 0; i < 8; i++) {
        ASSERT_EQ(Li[i], Li_cor[i]);
    }
    for(int i = 0; i < 8; i++) {
        ASSERT_FLOAT_EQ(Lx[i], Lx_cor[i]);
    }
    for(int i = 0; i < 7; i++) {
        ASSERT_EQ(Lnext[i], Next_cor[i]);
    }
    for(int i = 0; i < 7; i++) {
        ASSERT_EQ(Lprev[i], Prev_cor[i]);
    }
    for(int i = 0; i < 5; i++) {
        ASSERT_EQ(LPerm[i], Perm_cor[i]);
    }
}

class TestIgoFactor2 : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_A = nullptr;
    igo_sparse* igo_b = nullptr;
    igo_factor* igo_L = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        igo_A = igo_allocate_sparse(9, 9, 45, igo_cm);

        cholmod_sparse* A = igo_A->A;

        int* Ap = (int*) A->p;
        int* Ai = (int*) A->i;
        double* Ax = (double*) A->x;

        int Ap_setup[10] = {0, 3, 6, 9, 15, 21, 27, 33, 39, 45};
        int Ai_setup[45] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 
                            3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 3, 4, 5, 
                            6, 7, 8, 3, 4, 5, 6, 7, 8};
        double Ax_setup[45] = {1.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 
                               0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 
                               -0.999967, 0.008184, 0.033582, 0.000000, 1.000000, 0.000000, 
                               -0.008184, -0.999967, 0.999098, 0.000000, 0.000000, 1.000000, 
                               -0.000000, -0.000000, -1.000000, 1.000000, 0.000000, 0.000000, 
                               -0.999984, 0.005697, 0.017876, 0.000000, 1.000000, 0.000000, 
                               -0.005697, -0.999984, 1.003478, 0.000000, 0.000000, 1.000000, 
                               -0.000000, -0.000000, -1.000000};

        for(int i = 0; i < 10; i++) {
            Ap[i] = Ap_setup[i];
        }
        for(int i = 0; i < 45; i++) {
            Ai[i] = Ai_setup[i];
            Ax[i] = Ax_setup[i];
        }

        igo_b = igo_allocate_sparse(9, 1, 9, igo_cm);

        cholmod_sparse* b = igo_b->A;

        int* bp = (int*) b->p;
        int* bi = (int*) b->i;
        double* bx = (double*) b->x;

        int bp_setup[2] = {0, 9};
        int bi_setup[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        double bx_setup[9] = {0, 0, 0, -1.11022e-16, -0, 4.3644e-19, 
                              -2.22045e-16, 6.93889e-18, -1.30884e-18};

        for(int i = 0; i < 2; i++) {
            bp[i] = bp_setup[i];
        }
        for(int i = 0; i < 9; i++) {
            bi[i] = bi_setup[i];
            bx[i] = bx_setup[i];
        }

        igo_L = igo_allocate_factor(3, 3, igo_cm);
    }

    void TearDown() override {
        igo_free_sparse(&igo_A, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_sparse(&igo_b, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_factor(&igo_L, igo_cm);
        ASSERT_EQ(igo_L, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }
};

TEST_F(TestIgoFactor2, UpdownSolve) {
    igo_sparse* igo_Ab = igo_ssmult(igo_A, igo_b, igo_cm);
    cholmod_dense* Ab_dense = cholmod_sparse_to_dense(igo_Ab->A, igo_cm->cholmod_cm);
    igo_dense* igo_Ab_dense = igo_allocate_dense2(&Ab_dense, igo_cm);
    igo_dense* igo_y = igo_allocate_dense(3, 1, 3, igo_cm);

    igo_updown_solve(1, igo_A, igo_L, igo_y, igo_Ab_dense, igo_cm);

    igo_print_factor(3, "igo_L", igo_L, igo_cm);
    igo_print_dense(3, "igo_y", igo_y, igo_cm);

    double y_cor[9] = {-1.11022e-16, 0, 4.3643999999999997e-19, -1.6653583186297224e-16,
                       6.4845879760013512e-18, -3.5503565995572195e-18, 1.109738379035239e-16,
                       -2.9364858427188599e-18, 2.907999999977983e-19};

    ASSERT_EQ(igo_y->B->nrow, 9);
    ASSERT_EQ(igo_y->B->ncol, 1);

    double* yx = (double*) igo_y->B->x;
    for(int i = 0; i < 9; i++) {
        ASSERT_FLOAT_EQ(yx[i], y_cor[i]);
    }

    igo_free_sparse(&igo_Ab, igo_cm);
    igo_free_dense(&igo_Ab_dense, igo_cm);
    igo_free_dense(&igo_y, igo_cm);

}

class TestIgoFactor3 : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_factor* igo_L = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        igo_L = igo_allocate_factor(0, 0, igo_cm);
    }

    void TearDown() override {
        igo_free_factor(&igo_L, igo_cm);
        ASSERT_EQ(igo_L, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }
};

TEST_F(TestIgoFactor3, Resize1) {
    igo_resize_factor(10, 20, igo_L, igo_cm);
    igo_print_factor(3, "igo_L", igo_L, igo_cm);
}
