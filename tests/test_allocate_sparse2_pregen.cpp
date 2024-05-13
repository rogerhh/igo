
extern "C" {
#include "cholmod.h"
}

#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <iostream>

#include "igo.h"

using namespace std;

static const double double_eps = 1e-8;

class TestAllocateSparse2_FromFile : public ::testing::Test {
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

        igo_cm->solve_type = IGO_SOLVE_PCG;

        cholmod_triplet* A_tri = cholmod_read_triplet(stdin, igo_cm->cholmod_cm);
        cholmod_sparse* A = cholmod_triplet_to_sparse(A_tri, A_tri->nnz, igo_cm->cholmod_cm);
        cholmod_drop(0, A, igo_cm->cholmod_cm);
        A_orig = igo_allocate_sparse2(&A, igo_cm);

        cholmod_free_triplet(&A_tri, igo_cm->cholmod_cm);
    }

    static void TearDownTestSuite() {
        igo_free_sparse(&A_orig, igo_cm);
        A_orig = NULL;
        igo_finish(igo_cm);
        free(igo_cm);
        igo_cm = NULL;
    }

    void SetUp() override {
    }

    void TearDown() override {
    }
};

igo_common* TestAllocateSparse2_FromFile::igo_cm = NULL;
igo_sparse* TestAllocateSparse2_FromFile::A_orig = NULL;

TEST_F(TestAllocateSparse2_FromFile, test1) {

}
