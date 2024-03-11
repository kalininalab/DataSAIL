import logging
import shutil
import subprocess
import sys
from typing import Tuple, Optional
import platform

import cvxpy


def get_default(data_type: str, data_format: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Return the default clustering method for a specific type of data and a specific format.

    Args:
        data_type: Type of data as string representation
        data_format: Format encoded as string

    Returns:
        Tuple of the names of the method to use to compute either the similarity or distance for the input
    """
    if data_type == P_TYPE:
        if data_format == FORM_PDB:
            return FOLDSEEK, None
        elif data_format == FORM_FASTA:
            # Check if cd-hit is installed or neither of cd-hit and mmseqs are
            if INSTALLED[MMSEQS] or not INSTALLED[CDHIT]:
                return MMSEQS, None
            else:
                return CDHIT, None
    if data_type == M_TYPE and data_format == FORM_SMILES:
        return ECFP, None
    if data_type == G_TYPE:
        if data_format == FORM_FASTA:
            return CDHIT_EST, None
        elif data_format == FORM_GENOMES:
            return None, MASH
    return None, None


VERB_MAP = {
    "C": logging.CRITICAL,
    "F": logging.FATAL,
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
}

LOGGER = logging.getLogger("DataSAIL")

FORMATTER = logging.Formatter('%(asctime)s %(message)s')

_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(FORMATTER)
_stdout_handler.setLevel(logging.INFO)
LOGGER.addHandler(_stdout_handler)

if platform.system() == "Windows":
    from ctypes.wintypes import MAX_PATH
else:
    MAX_PATH = int(subprocess.check_output(["getconf", "PATH_MAX", "/"]))

# Define static values here, e.g. integer constants or string keys
WLK = "wlk"
MMSEQS = "mmseqs"
MMSEQS2 = "mmseqs2"
MMSEQSPP = "mmseqspp"
FOLDSEEK = "foldseek"
CDHIT = "cdhit"
CDHIT_EST = "cdhit_est"
ECFP = "ecfp"
MASH = "mash"
MASH_SKETCH = "mash_sketch"
MASH_DIST = "mash_dist"
TMALIGN = "tmalign"
SIM_ALGOS = [WLK, MMSEQS, MMSEQS2, MMSEQSPP, FOLDSEEK, CDHIT, CDHIT_EST, ECFP, ]
DIST_ALGOS = [MASH, ]
ALGOS = SIM_ALGOS + DIST_ALGOS
INSTALLED = {
    CDHIT: shutil.which("cd-hit") is not None,
    CDHIT_EST: shutil.which("cd-hit-est") is not None,
    MMSEQS: shutil.which("mmseqs") is not None,
    MMSEQS2: shutil.which("mmseqs") is not None,
    MMSEQSPP: shutil.which("mmseqs") is not None,
    MASH: shutil.which("mash") is not None,
    FOLDSEEK: shutil.which("foldseek") is not None,
    TMALIGN: shutil.which("TMalign") is not None,
}

UNK_LOCATION = "unknown"
P_TYPE = "P"
M_TYPE = "M"
G_TYPE = "G"
O_TYPE = "O"
FASTA_FORMATS = {"fasta", "fa", "fna"}
FORM_FASTA = "FASTA"
FORM_GENOMES = "Genomes"
FORM_OTHER = "Other"
FORM_PDB = "PDB"
FORM_SMILES = "SMILES"
NOT_ASSIGNED = "not selected"
MAX_CLUSTERS = 50

YAML_FILE_NAMES = {
    MMSEQS: "args/mmseqs2.yaml",
    MMSEQS2: "args/mmseqs2.yaml",
    MMSEQSPP: "args/mmseqspp.yaml",
    CDHIT: "args/cdhit.yaml",
    CDHIT_EST: "args/cdhit_est.yaml",
    FOLDSEEK: "args/foldseek.yaml",
    ECFP: "args/.yaml",
    MASH: "args/mash.yaml",
    MASH_SKETCH: "args/mash_sketch.yaml",
    MASH_DIST: "args/mash_dist.yaml",
}

KW_CACHE = "cache"
KW_CACHE_DIR = "cache_dir"
KW_CLI = "cli"
KW_DELTA = "delta"
KW_EPSILON = "epsilon"

KW_E_ARGS = "e_args"
KW_E_DATA = "e_data"
KW_E_DIST = "e_dist"
KW_E_ID_MAP = "e_id_map"
KW_E_WEIGHTS = "e_weights"
KW_E_STRAT = "e_strat"
KW_E_SIM = "e_sim"
KW_E_TYPE = "e_type"
KW_E_CLUSTERS = "e_clusters"

KW_F_ARGS = "f_args"
KW_F_DATA = "f_data"
KW_F_DIST = "f_dist"
KW_F_ID_MAP = "f_id_map"
KW_F_WEIGHTS = "f_weights"
KW_F_STRAT = "f_strat"
KW_F_SIM = "f_sim"
KW_F_TYPE = "f_type"
KW_F_CLUSTERS = "f_clusters"

KW_INTER = "inter"
KW_LOGDIR = "logdir"
KW_MAX_SEC = "max_sec"
KW_MAX_SOL = "max_sol"
KW_NAMES = "names"
KW_OUTDIR = "output"
KW_RUNS = "runs"
KW_SOLVER = "solver"
KW_SPLITS = "splits"
KW_TECHNIQUES = "techniques"
KW_THREADS = "threads"
KW_VERBOSE = "verbosity"
KW_LINKAGE = "linkage"

DEFAULT_KWARGS = {
    KW_CACHE: False,
    KW_CACHE_DIR: None,
    KW_CLI: False,
    KW_DELTA: 0.1,
    KW_EPSILON: 0.1,

    KW_E_ARGS: None,
    KW_E_CLUSTERS: 50,
    KW_E_DATA: None,
    KW_E_DIST: None,
    KW_E_ID_MAP: None,
    KW_E_WEIGHTS: None,
    KW_E_STRAT: None,
    KW_E_SIM: None,
    KW_E_TYPE: None,

    KW_F_ARGS: None,
    KW_F_CLUSTERS: 50,
    KW_F_DATA: None,
    KW_F_DIST: None,
    KW_F_ID_MAP: None,
    KW_F_WEIGHTS: None,
    KW_F_STRAT: None,
    KW_F_SIM: None,
    KW_F_TYPE: None,

    KW_INTER: None,
    KW_LOGDIR: None,
    KW_MAX_SEC: 1000,
    KW_MAX_SOL: 1000,
    KW_NAMES: ["train", "val", "test"],
    KW_OUTDIR: None,
    KW_RUNS: 1,
    KW_SOLVER: "SCIP",
    KW_SPLITS: [0.7, 0.2, 0.1],
    KW_TECHNIQUES: None,
    KW_THREADS: 1,
    KW_VERBOSE: "E",
    KW_LINKAGE: "average",
}

SOLVER_CBC = "CBC"
SOLVER_CPLEX = "CPLEX"
SOLVER_GLPK = "GLPK"
SOLVER_GLPK_MI = "GLPK_MI"
SOLVER_GUROBI = "GUROBI"
SOLVER_MOSEK = "MOSEK"
SOLVER_SCIP = "SCIP"
SOLVER_XPRESS = "XPRESS"
TEC_R = "R"
SRC_ID = "I"
SRC_CL = "C"
DIM_1 = "1"
DIM_2 = "2"
MODE_E = "e"
MODE_F = "f"
TEC_I1 = SRC_ID + DIM_1
TEC_C1 = SRC_CL + DIM_1
TEC_I2 = SRC_ID + DIM_2
TEC_C2 = SRC_CL + DIM_2

N_CLUSTERS = 10

# 0 Problem is mixed-integer ?!
# 1 Working
# 2 Hardly installable
# 3 Not installable
SOLVERS = {
    SOLVER_CBC: cvxpy.CBC,  # extra: CBC
    # "COPT": cvxpy.COPT,
    SOLVER_GLPK: cvxpy.GLPK_MI,
    SOLVER_GLPK_MI: cvxpy.GLPK_MI,
    SOLVER_SCIP: cvxpy.SCIP,
    SOLVER_CPLEX: cvxpy.CPLEX,
    SOLVER_GUROBI: cvxpy.GUROBI,
    SOLVER_MOSEK: cvxpy.MOSEK,
    SOLVER_XPRESS: cvxpy.XPRESS,
    # 0 "CLARABEL": cvxpy.CLARABEL,
    # 0 "GLOP": cvxpy.GLOP,
    # 0 "GLPK": cvxpy.GLPK,
    # 0 "OSQP": cvxpy.OSQP,
    # 0 "PROXQP": cvxpy.PROXQP,
    # 0 "PDLP": cvxpy.PDLP,
    # 0 "NAG": cvxpy.NAG,
    # 0 "ECOS": cvxpy.ECOS,
    # 0 "CVXOPT": cvxpy.CVXOPT,
    # 0 "SDPA": cvxpy.SDPA,
    # 0 "SCS": cvxpy.SCS,
    # 0 "SCIPY": cvxpy.SCIPY,
}
