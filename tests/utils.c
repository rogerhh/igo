#include "utils.h"

static unsigned long next = 1 ;

/* RAND_MAX assumed to be 32767 */
int my_rand (void)
{
   next = next * 1103515245 + 12345 ;
   return ((unsigned)(next/65536) % /* 32768 */ (MY_RAND_MAX + 1)) ;
}

void my_srand (unsigned seed)
{
   next = seed ;
}

unsigned long my_seed (void)
{
   return (next) ;
}

/* ========================================================================== */
/* === nrand ================================================================ */
/* ========================================================================== */

/* return a random Int between 0 and n-1 */

int nrand (int n)
{
    return ((n <= 0) ? (0) : (rand ( ) % n)) ;
}

/* ========================================================================== */
/* === xrand ================================================================ */
/* ========================================================================== */

/* return a random double between 0 and x */

double xrand (double range)
{
    return ((range * (double) (my_rand ( ))) / MY_RAND_MAX) ;
}


/* ========================================================================== */
/* === prand ================================================================ */
/* ========================================================================== */

/* allocate and construct a random permutation of 0:n-1 */

int *prand (int n)
{
    int *P ;
    int t, j, k ;
    P = (int*) malloc(n * sizeof(int)) ;
    for (k = 0 ; k < n ; k++)
    {
	P [k] = k ;
    }
    for (k = 0 ; k < n-1 ; k++)
    {
	j = k + nrand (n-k) ;
	t = P [j] ;
	P [j] = P [k] ;
	P [k] = t ;
    }
    return P;
}


/* ========================================================================== */
/* === rand_set ============================================================= */
/* ========================================================================== */

/* allocate and construct a random set of 0:n-1, possibly with duplicates */

int *rand_set (int len, int n)
{
    int *cset ;
    int k ;
    cset = (int*) malloc(len * sizeof(int));
    for (k = 0 ; k < len ; k++)
    {
	cset [k] = nrand (n) ;
    }
    return cset;
}

