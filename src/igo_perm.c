#include <igo.h>

/* ---------------------------------------------------------- */
/* Permutation functions */
/* ---------------------------------------------------------- */

igo_perm* igo_allocate_perm (
    /* --- input --- */
    int len,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_perm* P = (igo_perm*) malloc(sizeof(igo_perm));
    P->n_alloc = IGO_PERM_DEFAULT_N_ALLOC;
    while(P->n_alloc < len) {
        P->n_alloc *= 2;
    }
    P->P = (int*) malloc(P->n_alloc * sizeof(int));
    P->n = len;
    return P;
}

igo_perm* igo_allocate_perm2 (
    /* --- input --- */
    int len,
    int** P,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_perm* P1 = (igo_perm*) malloc(sizeof(igo_perm));
    P1->n_alloc = len;
    P1->n = len;
    P1->P = (*P);
    *P = NULL;
    return P1;
}

/* Perform P2 = P2 * P1
 * */
int igo_permute_permutation (
    /* --- input --- */
    int len,
    int* P1,
    /* --- in/out --- */
    int* P2,
    /* ------------- */
    igo_common* igo_cm
) {
    for(int i = 0; i < len; i++) {
        P2[i] = P1[P2[i]];
    }
    return 1;
}

/* Return P^(-1)
 * */
int* igo_invert_permutation (
    /* --- input --- */
    int len,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) {
    int* Pinv = (int*) malloc(len * sizeof(int));
    for(int i = 0; i < len; i++) {
        Pinv[P[i]] = i;
    }
    return Pinv;
}

/* Extend permutation to newlen. Assume P's memory is already allocated
 * */
int igo_extend_permutation (
    /* --- input --- */
    int Plen,
    int newlen,
    /* --- in/out --- */
    int* P,
    /* ------------- */
    igo_common* igo_cm
) {
    for(int i = Plen; i < newlen; i++) {
        P[i] = i;
    }
    return 1;
}

/* Extend permutation to newlen
 * */
int igo_extend_permutation2 (
    /* --- input --- */
    int newlen,
    /* --- in/out --- */
    igo_perm* P,
    /* ------------- */
    igo_common* igo_cm
) {
    if(P->n_alloc < newlen) {
        while(P->n_alloc < newlen) {
            P->n_alloc *= 2;
        }
        P->P = realloc(P->P, P->n_alloc * sizeof(int));
    }
    igo_extend_permutation(P->n, newlen, P->P, igo_cm);
    P->n = newlen;
    return 1;
}

void igo_print_permutation (
    /* --- input --- */
    int len,
    int* P,
    /* ------------- */
    igo_common* igo_cm
) {
    for(int i = 0; i < len; i++) {
        printf("%d ", P[i]);
    }
    printf("\n");
}

void igo_print_permutation2 (
    /* --- input --- */
    igo_perm* P,
    /* ------------- */
    igo_common* igo_cm
) {
    igo_print_permutation(P->n, P->P, igo_cm);
}
