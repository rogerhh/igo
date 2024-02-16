#include <iostream>
#include <gtest/gtest.h>
#include <petsc.h>
#include <petscdmshell.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <igo.h>
#include <petscvec.h>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    PetscCall(PetscInitialize(&argc, &argv, (char*)0, nullptr));

    int res = RUN_ALL_TESTS();

    PetscCall(PetscFinalize());

    return res;
}

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
        igo_cm->solve_type = IGO_SOLVE_BATCH;

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

extern PetscErrorCode MatVec(KSP ksp, Mat A, Mat v, void* cxt);
extern PetscErrorCode RHS(KSP ksp, Vec b, void* cxt);
extern PetscErrorCode PCVec(PC pc, Vec x, Vec y);
extern PetscErrorCode DMCreateMatrix_shell(DM dm, Mat* A);

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

     // Create the KSP solver
    int m = igo_cm->A->A->nrow;
    int n = igo_cm->A->A->ncol;
    KSP ksp;
    PC pc;
    DM dm;
    cholmod_sparse* A_col = igo_cm->A->A;
    Mat A;
    printf("%d %d\n", m, n);
    for(int i = 0; i < 9; i++) {
        printf("%d\n", ((int*) A_col->p)[i]);
    }
    PetscCallVoid(MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, n, m, (PetscInt*) A_col->p, (PetscInt*) A_col->i, (PetscScalar*) A_col->x, &A));
    PetscCallVoid(MatSetFromOptions(A));
    PetscCallVoid(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallVoid(KSPSetTolerances(ksp, 1.e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCallVoid(KSPSetOperators(ksp, A, A));
    PetscCallVoid(KSPSetType(ksp, KSPLSQR)); // Example: Use Conjugate Gradient method
    PetscCallVoid(KSPGetPC(ksp, &pc));
    PetscCallVoid(PCSetType(pc, PCSHELL));
    PetscCallVoid(PCShellSetApply(pc, PCVec));
    PetscCallVoid(KSPSetFromOptions(ksp));

    printf("Here0\n");

    // Set the custom operators for the linear system and preconditioner
    printf("Here1\n");

    // Set other solver options as needed
    printf("Here2\n");

    // Set the RHS vector and the initial guess
    Vec rhs, sol;
    PetscCallVoid(VecCreate(PETSC_COMM_WORLD, &rhs));
    PetscCallVoid(VecSetSizes(rhs, PETSC_DECIDE, n));
    PetscCallVoid(VecSetFromOptions(rhs));
    PetscCallVoid(VecCreate(PETSC_COMM_WORLD, &sol));
    PetscCallVoid(VecSetSizes(sol, PETSC_DECIDE, m));
    PetscCallVoid(VecSetFromOptions(sol));
    printf("Here3\n");


    vector<int> nidx(n);
    for(int i = 0; i < n; i++) {
        nidx[i] = i;
    }
    PetscCallVoid(VecSetValues(rhs, n, nidx.data(), (PetscScalar*) igo_cm->b->B->x, INSERT_VALUES));
    PetscCallVoid(VecAssemblyBegin(rhs));
    PetscCallVoid(VecAssemblyEnd(rhs));
    printf("Here4\n");
    
    PetscCallVoid(VecView(rhs, PETSC_VIEWER_STDOUT_SELF));
    PetscCallVoid(VecView(sol, PETSC_VIEWER_STDOUT_SELF));

    // Set the RHS vector
    printf("Here5\n");

    // Solve the system
    PetscCallVoid(KSPSolve(ksp, rhs, sol));
    printf("Here6\n");

    PetscInt num_iter;
    KSPGetIterationNumber(ksp, &num_iter);

    cout << "# of CG iterations: " << num_iter << endl;

    // Retrieve the solution
    // KSPGetSolution(ksp, &sol);

    // Destroy the KSP solver and finalize PETSc
    KSPDestroy(&ksp);
    VecDestroy(&rhs);
    VecDestroy(&sol);

}

/* Declare routines for user-provided preconditioner */
PetscErrorCode MatVec(KSP ksp, Mat A, Mat v, void* cxt) {
    printf("In MatVec\n");
    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));
    // PetscCall(MatView(v, PETSC_VIEWER_STDOUT_SELF));
    exit(0);
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Declare routines for user-provided preconditioner */
PetscErrorCode RHS(KSP ksp, Vec b, void* cxt) {
    printf("In RHS\n");
    // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_SELF));
    // PetscCall(MatView(v, PETSC_VIEWER_STDOUT_SELF));
    exit(0);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCVec(PC pc, Vec x, Vec y) {
    // printf("In PCVec\n");
    PetscFunctionBeginUser;
    igo_common* igo_cm;
    PetscCall(PCShellGetContext(pc, &igo_cm));
    PetscCall(VecCopy(x, y));
    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));
    PetscCall(VecView(y, PETSC_VIEWER_STDOUT_SELF));
    // return PETSC_SUCCESS;
    // exit(0);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateMatrix_shell(DM dm, Mat* A) {
    printf("In DMCreateMatrix_shell\n");
    PetscCall(MatCreate(PETSC_COMM_WORLD, A));
    PetscCall(MatSetSizes(*A, 9, 9, 9, 9));
    PetscCall(MatSetFromOptions(*A));
    PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatView(*A, PETSC_VIEWER_STDOUT_SELF));
    PetscFunctionReturn(PETSC_SUCCESS);
}
