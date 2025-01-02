import os

#N_THREADS = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
#os.environ["OPENBLAS_MAX_THREADS"] = N_THREADS
#os.environ["GOTO_NUM_THREADS"] = N_THREADS
#os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["GRB_LICENSE_FILE"] = "/home/rjo21/gurobi_mickey.lic"

from datasail.sail import sail

if __name__ == '__main__':
    sail()
