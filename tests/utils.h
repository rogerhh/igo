#pragma once

#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdlib.h>

/* ========================================================================== */
/* === my_rand ============================================================== */
/* ========================================================================== */

/* The POSIX example of rand, duplicated here so that the same sequence will
 * be generated on different machines. */

#define MY_RAND_MAX 32767

/* RAND_MAX assumed to be 32767 */
int my_rand (void);
void my_srand (unsigned seed);
unsigned long my_seed (void);

/* ========================================================================== */
/* === nrand ================================================================ */
/* ========================================================================== */

/* return a random Int between 0 and n-1 */

int nrand (int n);

/* ========================================================================== */
/* === xrand ================================================================ */
/* ========================================================================== */

/* return a random double between 0 and x */

double xrand (double range);


/* ========================================================================== */
/* === prand ================================================================ */
/* ========================================================================== */

/* allocate and construct a random permutation of 0:n-1 */

int *prand (int n);


/* ========================================================================== */
/* === rand_set ============================================================= */
/* ========================================================================== */

/* allocate and construct a random set of 0:n-1, possibly with duplicates */

int *rand_set (int len, int n);

