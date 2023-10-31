import os
import sys

parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from igo import *

if __name__ == "__main__":
    igo = Igo()
    print(igo.A)
