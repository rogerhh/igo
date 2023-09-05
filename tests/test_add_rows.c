#include <cholmod.h>
#include <stdio.h>
#include "igo.h"

void resize_sparse(
    int nrow, 
    int ncol, 
    int nzmax,
    cholmod_sparse* A
) {
    A->p = realloc(A->p, (ncol + 1) * sizeof(int));
    A->i = realloc(A->i, nzmax * sizeof(int));
    A->x = realloc(A->x, nzmax * sizeof(double));

    int old_ncol = A->ncol;
    int* Ap = A->p;

    for(int i = old_ncol + 1; i <= ncol; i++) {
        Ap[i] = Ap[old_ncol];
    }

    A->nrow = nrow;
    A->ncol = ncol;
    A->nzmax = nzmax;

}

void resize_factor(
    /* --- input --- */
    int nnew,   /* New size of the factor */
    /* --- inout --- */
    cholmod_factor* L   /* Factor we want to resize */
) {
    int nold = L->n;
    L->n = nnew;
    L->minor = L->n;
    L->Perm = realloc(L->Perm, L->n * sizeof(int));
    L->IPerm = realloc(L->IPerm, L->n * sizeof(int));
    L->ColCount = realloc(L->ColCount, L->n * sizeof(int));

    int* LPerm = (int*) L->Perm;
    int* LColCount = (int*) L->ColCount;

    // New rows and cols use the natural permutation
    for(int j = nold; j < nnew; j++) {
        LPerm[j] = j;
        LColCount[j] = 1;
    }

    if(!L->is_super) {
        int nzmaxold = L->nzmax;
        L->nzmax += nnew - nold;

        L->p = realloc(L->p, (L->n + 1) * sizeof(int));
        L->i = realloc(L->i, L->nzmax * sizeof(int));
        L->x = realloc(L->x, L->nzmax * sizeof(double));
        L->nz = realloc(L->nz, L->n * sizeof(int));
        // L->next = realloc(L->next, (L->n + 2) * )

        int* Lp = (int*) L->p;
        int* Li = (int*) L->i;
        double* Lx = (double*) L->x;
        int* Lnz = (int*) L->nz;

        // Add 1e-7 * I to extend the diagonal
        for(int j = nold; j < nnew; j++) {
            Lp[j + 1] = Lp[j] + 1;
            Li[Lp[j]] = j;
            Lx[Lp[j]] = 1e-6;
            Lnz[j] = 1;
        }

        L->next = realloc(L->next, (L->n + 2) * sizeof(int));
        L->prev = realloc(L->prev, (L->n + 2) * sizeof(int));

        int* Lnext = (int*) L->next;
        int* Lprev = (int*) L->prev;

        int oldfirstcol = Lnext[nold + 1];
        int oldlastcol = Lprev[nold];

        Lnext[nnew + 1] = oldfirstcol;  /* Next col of the head should be the first col */
        Lprev[nnew + 1] = -1;           /* Prev col of the head should be -1 */

        Lnext[nnew] = -1;               /* Next col of the tail should be -1 */
        Lprev[nnew] = nnew - 1;         /* Prev col of the tail should be new last col */

        Lprev[oldfirstcol] = nnew + 1;  /* Prev col of the first col should be the head */
        Lnext[oldlastcol] = nold;       /* Next col of the old last col should be the first added col */

        for(int j = nold; j < nnew; j++) {
            Lnext[j] = j + 1;
            Lprev[j] = j - 1;
        }

        printf("Next = ");
        for(int j = 0; j < nnew + 2; j++) {
            printf("%d ", Lnext[j]);
        }
        printf("\n");
        printf("Prev = ");
        for(int j = 0; j < nnew + 2; j++) {
            printf("%d ", Lprev[j]);
        }
        printf("\n");
    }
    else {
        // All the new rows and columns are in the same supernode
        int nsuperold = L->nsuper;
        L->nsuper += 1;

        L->super = realloc(L->super, (L->nsuper + 1) * sizeof(int));
        L->pi = realloc(L->pi, L->nsuper * sizeof(int));
        L->px = realloc(L->px, L->nsuper * sizeof(int));
        L->s = realloc(L->s, L->nsuper * sizeof(int));

        int* Ls = (int*) L->s;
        int* Lpi = (int*) L->pi;
        int* Lpx = (int*) L->px;
        int* Lsuper = (int*) L->super;

        Lsuper[L->nsuper] = nnew;
        Lpi[L->nsuper] = Lpi[nsuperold];
        Lpx[L->nsuper] = Lpx[nsuperold];
        Ls[L->nsuper] = Ls[nsuperold];
    }
    printf("Done resizing\n");
}

int main() {
    int nrows = 3, ncols = 5, nzmax = 15;
    cholmod_common c ;
    cholmod_start(&c);
    cholmod_sparse* A = 
        cholmod_allocate_sparse(2, 2, 3, true, true, 0, CHOLMOD_REAL, &c);

    cholmod_dense* X = NULL;
    cholmod_dense* B;

    int* Ap = (int*) A->p;
    int* Ai = (int*) A->i;
    double* Ax = (double*) A->x;

    Ap[0] = 0;
    Ap[1] = 1;
    Ap[2] = 3;
    Ai[0] = 0;
    Ai[1] = 0;
    Ai[2] = 1;
    Ax[0] = 1;
    Ax[1] = 1;
    Ax[2] = 1;

    printf("Ap = ");
    for(int j = 0; j <= ncols; j++) {
        printf("%d ", Ap[j]);
    }
    printf("\n");

    printf("Ai = ");
    for(int i = 0; i < nzmax; i++) {
        printf("%d ", Ai[i]);
    }
    printf("\n");

    for(int i = 0; i < nzmax; i++) {
        printf("%d\n", Ai[i]);
    }

    igo_print_cholmod_sparse(true, "A", A, &c);

    cholmod_sparse* C = cholmod_aat(A, NULL, 0, 1, &c);

    igo_print_cholmod_sparse(true, "C", C, &c);

    c.final_ll = true;
    c.supernodal = CHOLMOD_SIMPLICIAL;
    cholmod_factor* L = cholmod_analyze(A, &c);
    int res = cholmod_factorize(A, L, &c);
    printf("res = %d\n", res);

    printf("L->itype = %d\n", L->itype);
    igo_print_cholmod_factor(true, "L", L, &c);

    printf("ColCount = ");
    for(int i = 0; i < L->n; i++) {
        printf("%d ", ((int*) L->ColCount)[i]);
    }
    printf("\n");

    printf("Adding cols to A\n");

    resize_sparse(3, 3, 5, A);

    igo_print_cholmod_sparse(true, "A", A, &c);

    Ap = A->p;
    Ai = A->i;
    Ax = A->x;

    Ap[3] = 5;
    Ai[3] = 1;
    Ai[4] = 2;
    Ax[3] = 1;
    Ax[4] = 1;

    igo_print_cholmod_sparse(true, "A", A, &c);

    cholmod_sparse* Ahat = 
        cholmod_allocate_sparse(3, 1, 2, true, true, 0, CHOLMOD_REAL, &c);
    int* Ahatp = (int*) Ahat->p;
    int* Ahati = (int*) Ahat->i;
    double* Ahatx = (double*) Ahat->x;

    Ahatp[0] = 0;
    Ahatp[1] = 2;
    Ahati[0] = 1;
    Ahati[1] = 2;
    Ahatx[0] = 1;
    Ahatx[1] = 1;

    igo_print_cholmod_sparse(true, "Ahat", Ahat, &c);

    printf("Resizing factor L\n");
    resize_factor(3, L);

    igo_print_cholmod_factor(3, "L", L, &c);

    cholmod_updown(1, Ahat, L, &c);
    igo_print_cholmod_factor(3, "L", L, &c);
    cholmod_change_factor(CHOLMOD_REAL, 1, 0, 1, 1, L, &c);
    igo_print_cholmod_factor(3, "L", L, &c);
    
    cholmod_free_sparse(&A, &c);
    cholmod_finish(&c);

}
