import logging
import shutil
import sys

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

# Define static values here, e.g. integer constants or string keys
WLK = "wlk"
MMSEQS = "mmseqs"
MMSEQS2 = "mmseqs2"
FOLDSEEK = "foldseek"
CDHIT = "cdhit"
ECFP = "ecfp"
MASH = "mash"
MASH_SKETCH = "mash_sketch"
MASH_DIST = "mash"
TMALIGN = "tmalign"
SIM_ALGOS = [WLK, MMSEQS, MMSEQS2, FOLDSEEK, CDHIT, ECFP, ]
DIST_ALGOS = [MASH, ]
INSTALLED = {
    CDHIT: shutil.which("cd-hit") is not None,
    MMSEQS: shutil.which("mmseqs") is not None,
    MMSEQS2: shutil.which("mmseqs") is not None,
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
FORM_OTHER = "Other"
FORM_PDB = "PDB"
FORM_SMILES = "SMILES"
NOT_ASSIGNED = "not selected"

YAML_FILE_NAMES = {
    MMSEQS: "args/mmseqs2.yaml",
    MMSEQS2: "args/mmseqs2.yaml",
    CDHIT: "args/cdhit.yaml",
    FOLDSEEK: "args/foldseek.yaml",
    ECFP: "args/.yaml",
    MASH_SKETCH: "args/mash_sketch.yaml",
    MASH_DIST: "args/mash_dist.yaml",
}

KW_CACHE = "cache"
KW_CACHE_DIR = "cache_dir"
KW_CLI = "cli"
KW_DATA = "data"
KW_EPSILON = "epsilon"

KW_E_ARGS = "e_args"
KW_E_DATA = "e_data"
KW_E_DIST = "e_dist"
KW_E_ID_MAP = "e_id_map"
KW_E_MAX_SIM = "e_max_sim"
KW_E_MAX_DIST = "e_max_dist"
KW_E_WEIGHTS = "e_weights"
KW_E_SIM = "e_sim"
KW_E_TYPE = "e_type"

KW_F_ARGS = "f_args"
KW_F_DATA = "f_data"
KW_F_DIST = "f_dist"
KW_F_ID_MAP = "f_id_map"
KW_F_MAX_SIM = "f_max_sim"
KW_F_MAX_DIST = "f_max_dist"
KW_F_WEIGHTS = "f_weights"
KW_F_SIM = "f_sim"
KW_F_TYPE = "f_type"

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
KW_VECTORIZED = "vectorized"
KW_VERBOSE = "verbosity"

SOLVER_MOSEK = "MOSEK"
SOLVER_SCIP = "SCIP"
TEC_R = "R"
TEC_ICS = "ICS"
TEC_CCS = "CCS"
TEC_ICD = "ICD"
TEC_CCD = "CCD"
MODE_E = "e"
MODE_F = "f"
