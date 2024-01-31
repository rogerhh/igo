"""
gen_cholmod_update2_matrix.py
Generate 3 matrices & 2 vectors in triplet format for cholmod_updown2 testing.
Generate A, C, D where nonzero(C) = nonzero(D) = nonzero(AS), where S is a column selection matrix
A is height x width. width must > height to guarantee full row rank
To further ensure full row rank, A[0:height,0:height] = ridge * I
S is guaranteed to not select the first height columns to update
Additionally generate b, b' of size h x 1. b'_i != 0 iff column i is selected in S
"""

import sys
from optparse import OptionParser
import random
from copy import deepcopy
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

def print_triplet(fout, A):
    fout.write("%%MatrixMarket matrix coordinate real general\n")
    fout.write(f"{A.shape[0]} {A.shape[1]} {len(A.data)} \n")
    for j in range(A.shape[1]):
        col = A.getcol(j)
        for i in col.nonzero()[0]:
            fout.write(f"{i} {j} {col[i, 0]}\n")
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
    parser.add_option("--ridge", dest="ridge", type=float,
                      default=0.01, help="ridge constant to guarantee full row rank")
    parser.add_option("--scale", dest="scale", type=float,
                      default=5, help="Scale factor of generated matrix")
    parser.add_option("--min_update_col", dest="min_update_col", type=int,
                      default=1, help="Minimum number of updated columns")
    parser.add_option("--max_update_col", dest="max_update_col", type=int,
                      default=5, help="Maximum number of updated columns")
    parser.add_option("--seed", dest="seed", type=int,
                      default=None, help="Random seed for numpy sampling")
    parser.add_option("--outfile", dest="outfile",
                      default=None, help="Output file path")
    (options, args) = parser.parse_args()

    h = options.height
    w = options.width
    ridge = options.ridge
    scale = options.scale
    max_update_col = options.max_update_col
    min_col_nz = options.min_col_nz
    max_col_nz = options.max_col_nz
    min_update_col = options.min_update_col
    max_update_col = options.max_update_col

    assert(w > h)
    assert(max_col_nz <= h)
    assert(max_update_col <= w - h)

    if options.seed is not None:
        np.random.seed(options.seed)

    rows = []
    cols = []
    data = []

    # Generate identity matrix
    rows = [i for i in range(h)]
    cols = [i for i in range(h)]
    data = [ridge for i in range(h)]

    # Generate second half of A
    for c in range(h, w):
        col_nz = np.random.randint(min_col_nz, max_col_nz + 1)
        sel_rows = sorted(np.random.choice(h, size=col_nz, replace=False))
        rows.extend(sel_rows)
        cols.extend([c for _ in range(col_nz)])
        data.extend(scale * np.random.random(size=(col_nz,)))

    A = csc_matrix((data, (rows, cols)))

    # Generate C, which is a selection of columns from the second half of A
    # but with different numerical entries
    num_sel_col = np.random.randint(min_update_col, max_update_col + 1)
    sel_cols = sorted(np.random.choice(range(h, w), size=num_sel_col, replace=False))

    C = csc_matrix((h, w))
    D = csc_matrix((h, w))
    C[:, sel_cols] = A[:, sel_cols]
    D[:, sel_cols] = A[:, sel_cols]
    C.data = scale * np.random.random(size=C.data.shape)

    # Generate Atb
    sel_rows = sorted(list(set(C.nonzero()[0])))

    Atb = scale * np.random.random(size=(h, 1))
    delta_Atb = np.zeros((h, 1))
    delta_Atb[sel_rows, 0] = scale * np.random.random(size=(len(sel_rows,)))

    print(Atb)
    print(delta_Atb)


    with open(options.outfile, "w") as fout:
        print_triplet(fout, A)
        print_triplet(fout, C)
        print_triplet(fout, D)
        print_dense(fout, Atb)
        print_dense(fout, delta_Atb)
