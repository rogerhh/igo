#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>

#include "igo.h"


TEST(TestIgoUpdown, update1) {
    igo_common* igo_cm = (igo_common*) malloc(sizeof(igo_common));
    igo_init(igo_cm);

    igo_sparse* igo_A = igo_allocate_sparse(2, 3, 6, igo_cm);

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
    Ax[4] = 4;
    Ax[5] = 5;

    igo_dense* sol = igo_allocate_dense(2, 1, 2, igo_cm);

    double* solx = (double*) sol->B->x;

    solx[0] = 1;
    solx[1] = 1;

    double alpha_one[2] = {1, 1};
    double alpha_zero[2] = {0, 0};
    igo_dense* ATx = igo_allocate_dense(3, 1, 3, igo_cm);
    igo_dense* igo_Ab = igo_allocate_dense(2, 1, 2, igo_cm);
    igo_sdmult(igo_A, 1, alpha_one, alpha_zero, sol, ATx, igo_cm);
    igo_sdmult(igo_A, 0, alpha_one, alpha_zero, ATx, igo_Ab, igo_cm);

    igo_pcg_context* cxt = (igo_pcg_context*) malloc(sizeof(igo_pcg_context));
    igo_dense* igo_x = igo_allocate_dense(2, 1, 2, igo_cm);
    igo_print_dense(3, "igo_x before", igo_x, igo_cm);
    igo_factor* igo_L = igo_allocate_identity_factor(2, 5, 1, igo_cm);
    igo_solve_pcgne(igo_A, NULL, igo_Ab, igo_L, 1e-8, 1e-8, 10, igo_x, cxt, igo_cm);

    igo_print_dense(3, "igo_x after ", igo_x, igo_cm);

    printf("Num iter: %d\n", cxt->num_iter);
    printf("\n\n=======================================\n\n");

    int* Lp = (int*) igo_L->L->p;
    int* Li = (int*) igo_L->L->i;
    int* Lnz = (int*) igo_L->L->nz;
    double* Lx = (double*) igo_L->L->x;
    Lnz[Lp[0]] = 2;
    Li[Lp[0] + 1] = 1;
    Lx[Lp[0]] = 21;
    Lx[Lp[0] + 1] = 1.2857;
    Lx[Lp[1]] = 0.2857;

    double* Xx = (double*) igo_x->B->x;
    Xx[0] = 0;
    Xx[1] = 0;

    igo_solve_pcgne(igo_A, NULL, igo_Ab, igo_L, 1e-8, 1e-8, 10, igo_x, cxt, igo_cm);

    igo_print_dense(3, "igo_x after ", igo_x, igo_cm);

    printf("Num iter: %d\n", cxt->num_iter);

    free(cxt);
    igo_free_factor(&igo_L, igo_cm);
    igo_free_dense(&igo_x, igo_cm);
    igo_free_dense(&sol, igo_cm);
    igo_free_dense(&ATx, igo_cm);
    igo_free_dense(&igo_Ab, igo_cm);
    igo_free_sparse(&igo_A, igo_cm);

    igo_finish(igo_cm);
    free(igo_cm);
    igo_cm = NULL;
}

class TestIgoPcgne : public ::testing::Test {
public:
    igo_common* igo_cm = nullptr;
    igo_sparse* igo_A = nullptr;
    igo_sparse* igo_A_neg = nullptr;
    igo_dense* igo_Ab = nullptr;
    igo_dense* igo_x = nullptr;
    igo_dense* igo_sol = nullptr;

    void SetUp() override {
        igo_cm = (igo_common*) malloc(sizeof(igo_common));
        igo_init(igo_cm);

        cholmod_sparse* A = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, A, igo_cm->cholmod_cm);
        igo_A = igo_allocate_sparse2(&A, igo_cm);
        cholmod_sparse* A_neg = cholmod_read_sparse(stdin, igo_cm->cholmod_cm);
        cholmod_drop(0, A_neg, igo_cm->cholmod_cm);
        igo_A_neg = igo_allocate_sparse2(&A_neg, igo_cm);
        cholmod_dense* Ab = cholmod_read_dense(stdin, igo_cm->cholmod_cm);
        igo_Ab = igo_allocate_dense2(&Ab, igo_cm);
        cholmod_dense* sol = cholmod_read_dense(stdin, igo_cm->cholmod_cm);
        igo_sol = igo_allocate_dense2(&sol, igo_cm);
    }

    void TearDown() override {
        igo_free_sparse(&igo_A, igo_cm);
        ASSERT_EQ(igo_A, nullptr);

        igo_free_sparse(&igo_A_neg, igo_cm);
        ASSERT_EQ(igo_A_neg, nullptr);

        igo_free_dense(&igo_Ab, igo_cm);
        ASSERT_EQ(igo_Ab, nullptr);

        igo_free_dense(&igo_sol, igo_cm);
        ASSERT_EQ(igo_sol, nullptr);

        igo_free_dense(&igo_x, igo_cm);
        ASSERT_EQ(igo_x, nullptr);

        igo_finish(igo_cm);
        igo_cm = nullptr;
    }

};

TEST_F(TestIgoPcgne, TestPCGSolver) {
    int n = igo_Ab->B->nrow;
    igo_factor* L = igo_allocate_identity_factor(n, n, 1, igo_cm);
    igo_x = igo_allocate_dense(n, 1, n, igo_cm);
    igo_pcg_context* cxt = (igo_pcg_context*) malloc(sizeof(igo_pcg_context));
    igo_solve_pcgne(igo_A, igo_A_neg, igo_Ab, L, 1e-10, 1e-10, n + 10, igo_x, cxt, igo_cm);
    printf("num iter = %d\n", cxt->num_iter);

    ASSERT_TRUE(igo_dense_eq(igo_x, igo_sol, 1e-8, igo_cm));
}

