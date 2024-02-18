"""
gen_pcgne_matrix.py
If gen_A_neg is not set
    Generate matrix A, vector b, vector x, which satifies AA^T x = b
Otherwise 
    Generate matrix A, matrix A_neg, vector b, vector x, which satifies (AA^T - A_negA_neg^T) x = b
    AA^T - A_negA_neg^T is guaranteed to be positive definite

A is height x width. width must > height to guarantee full row rank
"""

import sys
from optparse import OptionParser
import random
from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

def print_triplet(fout, A):
    fout.write("%%MatrixMarket matrix coordinate real general\n")
    fout.write(f"{A.shape[0]} {A.shape[1]} {len(A.data) + 1} \n")
    # Adding a (0, 0) entry to indicate that the matrix is 1-based
    fout.write("0 0 0\n")
    for c in range(len(A.indptr) - 1):
        p = A.indptr[c]
        p1 = A.indptr[c + 1]
        for idx in range(p, p1):
            r = A.indices[idx]
            d = A.data[idx]
            fout.write(f"{r} {c} {d}\n")

    fout.write("\n")

def print_dense(fout, A):
    fout.write(f"{A.shape[0]} {A.shape[1]}\n")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            fout.write(f"{A[i, j]} ")
        fout.write("\n")
    fout.write("\n")

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--height", dest="height", type=int,
                      default=5, help="Height of the generated A matrix")
    parser.add_option("--width", dest="width", type=int,
                      default=10, help="Width of the generated A matrix")
    parser.add_option("--min_col_nz", dest="min_col_nz", type=int,
                      default=2, help="Minimum nonzero entries per column")
    parser.add_option("--max_col_nz", dest="max_col_nz", type=int,
                      default=3, help="Maximum nonzero entries per column")
    parser.add_option("--scale", dest="scale", type=float,
                      default=5, help="Scale factor of generated matrix")
    parser.add_option("--gen_A_neg", dest="gen_A_neg", 
                      action="store_true", help="Scale factor of generated matrix")
    parser.add_option("--seed", dest="seed", type=int,
                      default=None, help="Random seed for numpy sampling")
    parser.add_option("--outfile", dest="outfile",
                      default=None, help="Output file path")
    (options, args) = parser.parse_args()

    h = options.height
    w = options.width
    scale = options.scale
    min_col_nz = options.min_col_nz
    max_col_nz = options.max_col_nz
    outfile = options.outfile

    assert(w > h)
    assert(max_col_nz <= h)

    if options.seed is not None:
        np.random.seed(options.seed)

    rows = []
    cols = []
    data = []

    # # Generate identity matrix
    # rows = [i for i in range(h)]
    # cols = [i for i in range(h)]
    # data = [ridge for i in range(h)]

    # Generate second half of A
    for c in range(0, w):
        col_nz = np.random.randint(min_col_nz, max_col_nz + 1)
        sel_rows = sorted(np.random.choice(h, size=col_nz, replace=False))
        rows.extend(sel_rows)
        cols.extend([c for _ in range(col_nz)])
        data.extend(scale * np.random.random(size=(col_nz,)))

    A = csc_matrix((data, (rows, cols)))

    b = csc_matrix((h, 1))
    b[:, 0] = scale * np.random.random(size=(h, 1))

    print(A)

    if options.gen_A_neg:
        A_neg = -0.5 * A
    else:
        A_neg = csc_matrix(([], ([], [])))

    print(A_neg)

    H = A @ A.T

    H = H - A_neg @ A_neg.T

    sol = np.linalg.inv(H.A) @ b

    print(H @ sol - b)

    
    with open(outfile, "w") as fout:
        print_triplet(fout, A)
        print_triplet(fout, A_neg)
        print_dense(fout, b)
        print_dense(fout, sol)
    


