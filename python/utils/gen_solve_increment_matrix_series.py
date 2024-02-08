"""
gen_solve_incrmeent_matrix.py
Generate 3 matrices & 3 vectors in triplet format for cholmod_updown2 testing.
Generate A, A_hat, A_tilde & b, b_hat, b_tilde
A is height x width. width must > height to guarantee full row rank
To further ensure full row rank, A[0:height,0:height] = ridge * I
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
    parser.add_option("--ridge", dest="ridge", type=float,
                      default=0.01, help="ridge constant to guarantee full row rank")
    parser.add_option("--scale", dest="scale", type=float,
                      default=5, help="Scale factor of generated matrix")
    parser.add_option("--min_update_col", dest="min_update_col", type=int,
                      default=1, help="Minimum number of A_tilde columns")
    parser.add_option("--max_update_col", dest="max_update_col", type=int,
                      default=5, help="Maximum number of A_tilde columns")
    parser.add_option("--min_obs_col", dest="min_obs_col", type=int,
                      default=1, help="Minimum number of A_hat columns")
    parser.add_option("--max_obs_col", dest="max_obs_col", type=int,
                      default=5, help="Maximum number of A_hat columns")
    parser.add_option("--min_new_row", dest="min_new_row", type=int,
                      default=1, help="Minimum number of new rows in A_hat")
    parser.add_option("--max_new_row", dest="max_new_row", type=int,
                      default=5, help="Maximum number of new rows in A_hat")
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
    min_obs_col = options.min_obs_col
    max_obs_col = options.max_obs_col
    min_new_row = options.min_new_row
    max_new_row = options.max_new_row

    assert(w > h)
    assert(max_col_nz <= h)
    assert(max_update_col <= w - h)

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

    b = csc_matrix((w, 1))
    b[:, 0] = scale * np.random.random(size=(w, 1))

    # Generate A_tilde, b_tilde
    num_sel_col = np.random.randint(min_update_col, max_update_col + 1)
    sel_cols = sorted(np.random.choice(range(w), size=num_sel_col, replace=False))

    A_tilde = csc_matrix((h, w))
    A_tilde[:, sel_cols] = A[:, sel_cols]
    A_tilde.data = scale * np.random.random(size=A_tilde.data.shape)

    sel_cols = sorted(list(set(sel_cols)))
    b_tilde = csc_matrix((w, 1))
    b_tilde[sel_cols, 0] = scale * np.random.random(size=len(sel_cols))

    # Generate A_hat, b_hat
    num_obs_col = np.random.randint(min_obs_col, max_obs_col + 1)
    num_new_row = np.random.randint(min_new_row, max_new_row + 1)

    # Generate identity matrix
    rows = []
    cols = []
    data = []
    bhat_cols = []
    bhat_data = []

    # Generate A_hat
    for c in range(num_obs_col):
        col_nz = np.random.randint(min_col_nz, max_col_nz + 1)
        sel_rows = sorted(np.random.choice(h + max_new_row, size=col_nz, replace=False))
        rows.extend(sel_rows)
        cols.extend([c for _ in range(col_nz)])
        data.extend(scale * np.random.random(size=(col_nz,)))
        bhat_cols.append(c)
        bhat_data.append(np.random.random())

    print(cols)
    A_hat = csc_matrix((data, (rows, cols)))
    b_hat = csc_matrix((bhat_data, (bhat_cols, [0 for _ in range(len(bhat_cols))])))

    with open(options.outfile, "w") as fout:
        print_triplet(fout, A)
        print_triplet(fout, b)
        print_triplet(fout, A_tilde)
        print_triplet(fout, b_tilde)
        print_triplet(fout, A_hat)
        print_triplet(fout, b_hat)
