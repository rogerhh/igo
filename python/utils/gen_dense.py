import numpy as np
import sys

gen_dense_specs = {
    "small": ((2, 8), (2, 8)),
    "large": ((100, 200), (100, 200)),
    "skewed-row": ((500, 600), (1, 10)),
    "skewed-col": ((1, 10), (500, 600))
}

def print_dense(fout, A):
    fout.write(f"{A.shape[0]} {A.shape[1]}\n")
    for i in range(A.shape[1]):
        for j in range(A.shape[0]):
            fout.write(f"{A[j, i]}\n")
    fout.write("\n")

def gen_dense(nrows, ncols):
    A = np.random.rand(nrows, ncols) * 100 - 50
    return A

def gen_dense_spec(spec):
    row_range, col_range = gen_dense_specs[spec]
    return gen_dense(np.random.randint(row_range[0], row_range[1]), np.random.randint(col_range[0], col_range[1]))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_dense.py <spec> <output file>")
        print("Available specs:")
        for n, _ in gen_dense_specs.items():
            print(n)
    elif sys.argv[1] == "resize":
        A = gen_dense_spec("large")
        x, y = A.shape
        B = np.zeros((x + np.random.randint(-20, 21), y + np.random.randint(-20, 21)))
        B[:min(x, B.shape[0]), :min(y, B.shape[1])] = A[:min(x, B.shape[0]), :min(y, B.shape[1])]
        with open(sys.argv[2], "w") as fout:
            print_dense(fout, A)
            print_dense(fout, B)
    else:
        A = gen_dense_spec(sys.argv[1])
        with open(sys.argv[2], "w") as fout:
            print_dense(fout, A)