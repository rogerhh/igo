#include <iostream>
#include <gtest/gtest.h>
#include <petsc.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>

using namespace std;

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    PetscCall(PetscInitialize(&argc, &argv, (char*)0, nullptr));

    int res = RUN_ALL_TESTS();

    PetscCall(PetscFinalize());

    return res;
}


TEST(PETSC_Simple, CreateVector1) {
    Vec x; /* vectors */
    int n = 10;
    PetscCallVoid(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCallVoid(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCallVoid(VecSetFromOptions(x));
    PetscCallVoid(VecSet(x, 1.0));

    PetscScalar* a = (PetscScalar*) malloc(n * sizeof(PetscScalar));

    PetscCallVoid(VecGetArray(x, &a));

    for(int i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(a[i], 1);
    }
    PetscCallVoid(VecDestroy(&x));
}

TEST(PETSC_Simple, CreateSparseMatrix1) {
    Mat A; /* vectors */
    int n = 5;
    int m = 2;
    PetscCallVoid(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCallVoid(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCallVoid(MatSetFromOptions(A));

    PetscScalar v[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    PetscInt idxm[] = {0, 1};
    PetscInt idxn[] = {0, 1, 2, 3, 4};

    PetscCallVoid(MatSetValues(A, m, idxm, n, idxn, v, ADD_VALUES));

    PetscCallVoid(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscCallVoid(MatDestroy(&A));
}

TEST(PETSC_Simple, CreateSparseMatrix2) {
    Mat A; /* vectors */
    int n = 5;
    int m = 3;

    PetscScalar v[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    PetscInt i[] = {0, 3, 6, 10};
    PetscInt j[] = {0, 1, 2, 1, 2, 4, 0, 2, 3, 4};

    PetscCallVoid(MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, m, n, i, j, v, &A));
    PetscCallVoid(MatSetFromOptions(A));

    PetscCallVoid(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscCallVoid(MatDestroy(&A));

}

TEST(PETSC_Simple, SolveDirect) {
    Mat A; 
    Vec x, b, u; /* approx solution, RHS, exact solution */
    KSP ksp;
    PC pc;      /* preconditioner context */
    int n = 5;
    int m = 5;

    PetscScalar v[] = {1, 2, 3, 
                       4, 5, 6, 
                       7, 8, 9, 10,
                       3, 4,
                       5, 6};
    PetscInt i[] = {0, 3, 6, 10, 12, 14};
    PetscInt j[] = {0, 1, 2, 
                    1, 2, 4, 
                    0, 2, 3, 4,
                    2, 3,
                    0, 4};

    PetscCallVoid(MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, m, n, i, j, v, &A));
    PetscCallVoid(MatSetFromOptions(A));

    PetscCallVoid(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

    PetscCallVoid(MatView(A, PETSC_VIEWER_STDOUT_SELF));

    PetscCallVoid(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCallVoid(PetscObjectSetName((PetscObject)x, "Solution"));
    PetscCallVoid(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCallVoid(VecSetFromOptions(x));
    PetscCallVoid(VecDuplicate(x, &b));
    PetscCallVoid(VecDuplicate(x, &u));

    PetscCallVoid(VecSet(u, 1.0));
    PetscCallVoid(MatMult(A, u, b));

    PetscCallVoid(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCallVoid(KSPSetOperators(ksp, A, A));
    PetscCallVoid(KSPSetType(ksp, KSPCGNE));

    PetscCallVoid(KSPGetPC(ksp, &pc));
    PetscCallVoid(PCSetType(pc, PCJACOBI));
    PetscCallVoid(KSPSetTolerances(ksp, 1.e-10, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCallVoid(KSPSetFromOptions(ksp));

    PetscCallVoid(KSPSolve(ksp, b, x)); 

    PetscInt num_iter;
    KSPGetIterationNumber(ksp, &num_iter);

    cout << "# of CG iterations: " << num_iter << endl;
    PetscCallVoid(VecView(b, PETSC_VIEWER_STDOUT_SELF));
    PetscCallVoid(VecView(x, PETSC_VIEWER_STDOUT_SELF));

    PetscCallVoid(MatDestroy(&A));
    PetscCallVoid(VecDestroy(&x));
    PetscCallVoid(VecDestroy(&u));
    PetscCallVoid(VecDestroy(&b));
    PetscCallVoid(KSPDestroy(&ksp));

}
