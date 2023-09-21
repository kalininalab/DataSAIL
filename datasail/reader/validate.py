import os
from argparse import Namespace
from typing import Tuple, Union, Optional

from datasail.parsers import MultiYAMLParser
from datasail.settings import CDHIT, MMSEQS2, MASH, MASH_SKETCH, MASH_DIST, FOLDSEEK, MMSEQS, get_default


def validate_user_args(
        dtype: str,
        dformat: str,
        similarity: str,
        distance: str,
        tool_args: str,
) -> Optional[Union[Namespace, Tuple[Optional[Namespace], Optional[Namespace]]]]:
    """
    Validate the arguments from the user for an external clustering program.

    Args:
        dtype: Type of data to be clustered
        dformat: Format of the data to be clustered
        similarity: similarity to be used for clustering
        distance: distance to be used for clustering
        tool_args: Arguments to be passed to the external clustering program

    Returns:
        The namespace containing the parsed and validated arguments
    """
    sim_none, dist_none = similarity is None, distance is None
    both_none = sim_none and dist_none
    if (not sim_none and similarity.lower() == CDHIT) or (both_none and get_default(dtype, dformat)[0] == CDHIT):
        return check_cdhit_arguments(tool_args)
    elif (not sim_none and similarity.lower()[:6] == MMSEQS) or (
            both_none and get_default(dtype, dformat)[0] == MMSEQS2):
        return check_mmseqs_arguments(tool_args)
    elif (not sim_none and similarity.lower() == FOLDSEEK) or (
            both_none and get_default(dtype, dformat)[0] == FOLDSEEK):
        return check_mash_arguments(tool_args)
    elif (not dist_none and distance.lower() == MASH) or (both_none and get_default(dtype, dformat)[1] == MASH):
        return check_mash_arguments(tool_args)
    else:
        return None


def check_cdhit_arguments(args: str = "") -> Namespace:
    """
    Validate the custom arguments provided to DataSAIL for executing MASH.

    Args:
        args: String of the arguments that can be set by user
    """
    # args = args.split(" ") if " " in args else (args if isinstance(args, list) else [args])
    args = MultiYAMLParser(CDHIT).parse_args(args)
    # Check if -c, -s, -aL, -aS, -uL, -uS values are within the valid range
    if not (0 <= args.c <= 1):
        raise ValueError("Invalid value for -c. It should be between 0 and 1.")
    if not (0 <= args.s <= 1):
        raise ValueError("Invalid value for -s. It should be between 0 and 1.")
    if not (0 <= args.aL <= 1):
        raise ValueError("Invalid value for -aL. It should be between 0 and 1.")
    if not (0 <= args.aS <= 1):
        raise ValueError("Invalid value for -aS. It should be between 0 and 1.")
    if not (0 <= args.uL <= 1):
        raise ValueError("Invalid value for -uL. It should be between 0 and 1.")
    if not (0 <= args.uS <= 1):
        raise ValueError("Invalid value for -uS. It should be between 0 and 1.")

    # Check if -G, -p, -g, -sc, -sf, -bak values are either 0 or 1
    if args.G not in [0, 1]:
        raise ValueError("Invalid value for -G. It should be either 0 or 1.")
    # if args.p not in [0, 1]:
    #     raise ValueError("Invalid value for -p. It should be either 0 or 1.")
    if args.g not in [0, 1]:
        raise ValueError("Invalid value for -g. It should be either 0 or 1.")
    # if args.sc not in [0, 1]:
    #     raise ValueError("Invalid value for -sc. It should be either 0 or 1.")
    # if args.sf not in [0, 1]:
    #     raise ValueError("Invalid value for -sf. It should be either 0 or 1.")
    # if args.bak not in [0, 1]:
    #     raise ValueError("Invalid value for -bak. It should be either 0 or 1.")
    if not ((args.n == 2 and 0.4 <= args.c <= 0.5) or
            (args.n == 3 and 0.5 <= args.c <= 0.6) or
            (args.n == 4 and 0.6 <= args.c <= 0.7) or
            (args.n == 5 and 0.7 <= args.c <= 1.0)):
        raise ValueError("There are restrictions on the values for n and c in CD-HIT:\n"
                         "n == 5 <=> c in [0.7, 1.0]\n"
                         "n == 4 <=> c in [0.6, 0.7]\n"
                         "n == 3 <=> c in [0.5, 0.6]\n"
                         "n == 2 <=> c in [0.4, 0.5]")

    # Check other values are within the valid range
    if not (1 <= args.b <= 32):
        raise ValueError("Invalid value for -b. It should be between 1 and 32.")
    if not (0 <= args.M):
        raise ValueError("Invalid value for -M. It should be greater than or equal to 0.")
    if not (0 <= args.T):
        raise ValueError("Invalid value for -T. It should be greater than or equal to 0.")
    if not (0 <= args.n):
        raise ValueError("Invalid value for -n. It should be greater than or equal to 0.")
    # if not (0 <= args.l):
    #     raise ValueError("Invalid value for -l. It should be greater than or equal to 0.")
    if not (0 <= args.t):
        raise ValueError("Invalid value for -t. It should be greater than or equal to 0.")
    # if not (0 <= args.d):
    #     raise ValueError("Invalid value for -d. It should be greater than or equal to 0.")
    if not (0 <= args.S <= 4294967296):
        raise ValueError("Invalid value for -S. It should be between 0 and 4294967296.")
    if not (0 <= args.AL):
        raise ValueError("Invalid value for -AL. It should be greater than or equal to 0.")
    if not (0 <= args.AS):
        raise ValueError("Invalid value for -AS. It should be greater than or equal to 0.")
    if not (0 <= args.A):
        raise ValueError("Invalid value for -A. It should be greater than or equal to 0.")
    if not (0 <= args.U):
        raise ValueError("Invalid value for -U. It should be greater than or equal to 0.")

    if args.G == 0 and (args.aL != 0.0 or args.AL != 99999999):
        raise ValueError("Options -G 0 is incompatible with -aL and -AL.")
    if args.s != 0.0 and args.S != 999999:
        raise ValueError("Options -s and -S are incompatible.")
    if args.aL != 0.0 and args.AL != 99999999:
        raise ValueError("Options -aL and -AL are incompatible.")
    if args.aS != 0.0 and args.AS != 99999999:
        raise ValueError("Options -aS and -AS are incompatible.")
    if args.uL != 1.0 and (args.uS != 1.0 or args.U != 99999999):
        raise ValueError("Option -uL is incompatible with -uS and -U.")
    if args.A != 0 and (args.aL != 0.0 or args.AL != 99999999):
        raise ValueError("Option -A is incompatible with -aL and -AL.")
    if args.A != 0 and (args.aS != 0.0 or args.AS != 99999999):
        raise ValueError("Option -A is incompatible with -aS and -AS.")
    return args


def check_mmseqs_arguments(args: str = "") -> Optional[Namespace]:
    """
    Validate the custom arguments provided to DataSAIL for executing MMSEQS2.

    Args:
        args: String of the arguments that can be set by user

    Returns:
        The namespace containing the parsed and validated arguments.
    """
    # Reference: https://github.com/soedinglab/MMseqs2/blob/master/src/commons/Parameters.cpp
    args = MultiYAMLParser(MMSEQS2).parse_args(args)

    # Define a function to check valid range
    def check_valid_range(value, min_val, max_val, name):
        if not (min_val <= value <= max_val):
            raise ValueError(f"Invalid value for {name}. It should be between {min_val} and {max_val}.")

    def check_valid_set(value, min_val, max_val, name):
        if not (isinstance(value, int) and min_val <= value < max_val):
            raise ValueError(f"Invalid value for {name}. It should be an integer between {min_val} and {max_val}.")

    # Check numeric value ranges
    check_valid_range(args.s, 1.0, 7.5, "-s")
    check_valid_range(args.comp_bias_corr, 0, 1, "--comp-bias-corr")
    check_valid_range(args.exact_kmer_matching, 0, 1, "--exact-kmer-matching")
    check_valid_range(args.mask_prob, 0, 1, "--mask-prob")
    check_valid_range(args.min_ungapped_score, 0, float('inf'), "--min-ungapped-score")
    check_valid_range(args.c, 0, 1, "-c")
    check_valid_range(args.e, 0, float('inf'), "-e")
    check_valid_range(args.min_seq_id, 0, 1, "--min-seq-id")

    check_valid_set(args.seq_id_mode, 0, 3, "--seq-id-mode")
    check_valid_set(args.split_mode, 0, 3, "--split-mode")
    check_valid_set(args.cov_mode, 0, 6, "--cov-mode")
    check_valid_set(args.alignment_mode, 0, 6, "--alignment-mode")
    check_valid_set(args.cluster_mode, 0, 4, "--cluster-mode")
    check_valid_set(args.similarity_type, 1, 3, "--similarity-type")
    check_valid_set(args.rescore_mode, 0, 5, "--rescore-mode")
    check_valid_set(args.dbtype, 0, 3, "--dbtype")
    check_valid_set(args.createdb_mode, 0, 2, "--createdb-mode")
    check_valid_set(args.max_seq_len, 1, 65536, "--max-seq-len")
    # check_valid_set(args.v, 0, 4, "-v")
    check_valid_set(args.max_iterations, 1, 2147483647, "--max-iterations")
    check_valid_set(args.min_aln_len, 0, 2147483647, "--min-seq-id")

    # TODO: exact-kmer-matching range (0,1) or values 0 or 1?

    # Check boolean values
    # for arg_name in ["diag_score", "exact_kmer_matching", "mask", "mask_lower_case", "add_self_matches",
    #                  "wrapped_scoring", "realign", "cluster_reassign", "single_step_clustering",
    #                  "adjust_kmer_len", "ignore_multi_kmer"]:
    for arg_name in ["mask", "mask_lower_case", "spaced_kmer_mode", "sort_results"]:
        if getattr(args, arg_name) not in [0, 1]:
            raise ValueError(f"Invalid value for --{arg_name.replace('_', '-')}. It should be 0 or 1.")

    # Check integer values
    for arg_name in ["k", "max_seqs", "split", "threads", "zdrop", "id_offset",
                     "cluster_steps", "max_rejected", "max_accept", "realign_max_seqs", "min_aln_len", "hash_shift",
                     "kmer_per_seq"]:
        if not (0 <= getattr(args, arg_name) <= 2147483647):
            raise ValueError(f"Invalid value for --{arg_name.replace('_', '-')}. It should be a non-negative integer.")

    # Check floating-point values  # Add: "realign_score_bias" ?
    for arg_name in ["score_bias", "corr_score_weight", "corr_score_weight"]:
        if getattr(args, arg_name) < 0.0:
            raise ValueError(f"Invalid value for --{arg_name.replace('_', '-')}. It should be a non-negative float.")

    # Check string format values
    if args.k_score.count(',') != 1 or len(args.k_score.split(',')) != 2:
        raise ValueError("Invalid format for --k-score. It should be in the format 'seq:value,prof:value'.")
    # TODO: check for values (0 - 2147483647)

    if args.alph_size.count(',') != 1 or len(args.alph_size.split(',')) != 2:
        raise ValueError("Invalid format for --alph-size. It should be in the format 'aa:value,nucl:value'.")
    # TODO: check for values (2 - 21)

    if args.gap_open.count(',') != 1 or len(args.gap_open.split(',')) != 2:
        raise ValueError("Invalid format for --gap-open. It should be in the format 'aa:value,nucl:value'.")

    if args.gap_extend.count(',') != 1 or len(args.gap_extend.split(',')) != 2:
        raise ValueError("Invalid format for --gap-extend. It should be in the format 'aa:value,nucl:value'.")

    if args.sub_mat.count(',') != 1 or len(args.sub_mat.split(',')) != 2:
        raise ValueError("Invalid format for --sub-mat. It should be in the format 'aa:value,nucl:value'.")

    # Check custom conditions

    if args.kmer_per_seq_scale.count(',') != 1 or len(args.kmer_per_seq_scale.split(',')) != 2:
        raise ValueError("Invalid format for --kmer-per-seq-scale. It should be in the format 'aa:value,nucl:value'.")
    # TODO: Check values (range unknown, maybe float)

    # aa_kmer_per_seq_scale, nucl_kmer_per_seq_scale = map(float, args.kmer_per_seq_scale.split(','))
    # if aa_kmer_per_seq_scale <= 0 or nucl_kmer_per_seq_scale <= 0:
    #     raise ValueError("Invalid values for --kmer-per-seq-scale. Values should be positive floats.")

    return args


def check_foldseek_arguments(args: str = "") -> Namespace:
    """
    Validate the custom arguments provided to DataSAIL for executing FOLDSEEK.

    Args:
        args: Namespace object containing parsed arguments

    Returns:
        args: Validated Namespace object
    """
    args = MultiYAMLParser(FOLDSEEK).parse_args(args)

    if not (0 <= args.comp_bias_corr <= 1):
        raise ValueError("Invalid value for comp_bias_corr. It should be between 0 and 1.")

    if not (0.0 <= args.comp_bias_corr_scale <= 1.0):
        raise ValueError("Invalid value for comp_bias_corr_scale. It should be between 0.0 and 1.0.")

    if not (1.0 <= args.s <= 7.5):
        raise ValueError("Invalid value for s. It should be between 1.0 and 7.5.")

    if not (0 <= args.k <= 99):
        raise ValueError("Invalid value for k. It should be between 0 and 99.")

    if not (0 <= args.max_seqs <= 2147483647):
        raise ValueError("Invalid value for max_seqs. It should be between 0 and 2147483647.")

    if not (0 <= args.split <= 2147483647):
        raise ValueError("Invalid value for split. It should be between 0 and 2147483647.")

    if not (0 <= args.split_mode <= 2):
        raise ValueError("Invalid value for split_mode. It should be between 0 and 2.")

    if not (0 <= args.diag_score <= 1):
        raise ValueError("Invalid value for diag_score. It should be either 0 or 1.")

    if not (0 <= args.exact_kmer_matching <= 1):
        raise ValueError("Invalid value for exact_kmer_matching. It should be either 0 or 1.")

    if not (0 <= args.mask <= 1):
        raise ValueError("Invalid value for mask. It should be either 0 or 1.")

    if not (0.0 <= args.mask_prob <= 1.0):
        raise ValueError("Invalid value for mask_prob. It should be between 0.0 and 1.0.")

    if not (0 <= args.mask_lower_case <= 1):
        raise ValueError("Invalid value for mask_lower_case. It should be either 0 or 1.")

    if not (0 <= args.min_ungapped_score <= 2147483647):
        raise ValueError("Invalid value for min_ungapped_score. It should be between 0 and 2147483647.")

    if not (0 <= args.spaced_kmer_mode <= 1):
        raise ValueError("Invalid value for spaced_kmer_mode. It should be either 0 or 1.")

    if not (0 <= args.alignment_mode <= 3):
        raise ValueError("Invalid value for alignment_mode. It should be between 0 and 3.")

    if not (0 <= args.alignment_output_mode <= 5):
        raise ValueError("Invalid value for alignment_output_mode. It should be between 0 and 5.")

    # if not (0.0 <= args.e <= float("inf")):
    #     raise ValueError("Invalid value for e. It should be between 0.0 and infinity.")

    if not (0 <= args.min_aln_len <= 2147483647):
        raise ValueError("Invalid value for min_aln_len. It should be between 0 and 2147483647.")

    if not (0 <= args.seq_id_mode <= 2):
        raise ValueError("Invalid value for seq_id_mode. It should be between 0 and 2.")

    # if not (0 <= args.alt_ali <= 2147483647):
    #     raise ValueError("Invalid value for alt_ali. It should be between 0 and 2147483647.")

    if not (0 <= args.num_iterations <= 2147483647):
        raise ValueError("Invalid value for num_iterations. It should be between 0 and 2147483647.")

    if not (0.0 <= args.tmscore_threshold <= 1.0):
        raise ValueError("Invalid value for tmscore_threshold. It should be between 0.0 and 1.0.")

    if not (0 <= args.tmalign_hit_order <= 4):
        raise ValueError("Invalid value for tmalign_hit_order. It should be between 0 and 4.")

    if not (0 <= args.tmalign_fast <= 1):
        raise ValueError("Invalid value for tmalign_fast. It should be either 0 or 1.")

    if not (0.0 <= args.lddt_threshold <= 1.0):
        raise ValueError("Invalid value for lddt_threshold. It should be between 0.0 and 1.0.")

    if not (0 <= args.prefilter_mode <= 2):
        raise ValueError("Invalid value for prefilter_mode. It should be between 0 and 2.")

    if not (0 <= args.alignment_type <= 2):
        raise ValueError("Invalid value for alignment_type. It should be between 0 and 2.")

    if not (0 <= args.cluster_search <= 1):
        raise ValueError("Invalid value for cluster_search. It should be either 0 or 1.")

    if not (0 <= args.mask_bfactor_threshold <= 100):
        raise ValueError("Invalid value for mask_bfactor_threshold. It should be between 0 and 100.")

    # if not (0 <= args.format_mode <= 5):
    #     raise ValueError("Invalid value for format_mode. It should be between 0 and 5.")

    if not (0 <= args.greedy_best_hits <= 1):
        raise ValueError("Invalid value for greedy_best_hits. It should be either 0 or 1.")

    if not (0 <= args.db_load_mode <= 3):
        raise ValueError("Invalid value for db_load_mode. It should be between 0 and 3.")

    if not (1 <= args.threads <= 2147483647):
        raise ValueError("Invalid value for threads. It should be between 1 and 2147483647.")

    # if not (0 <= args.v <= 3):
    #     raise ValueError("Invalid value for v. It should be between 0 and 3.")

    if not (0 <= args.max_seq_len <= 65536):
        raise ValueError("Invalid value for max_seq_len. It should be between 0 and 65536.")

    # if not (0 <= args.compressed <= 1):
    #     raise ValueError("Invalid value for compressed. It should be either 0 or 1.")

    # if not (0 <= args.remove_tmp_files <= 1):
    #     raise ValueError("Invalid value for remove_tmp_files. It should be either 0 or 1.")

    # if not (0 <= args.force_reuse <= 1):
    #     raise ValueError("Invalid value for force_reuse. It should be either 0 or 1.")

    if not (0 <= args.zdrop <= 2147483647):
        raise ValueError("Invalid value for zdrop. It should be between 0 and 2147483647.")

    if not (0 <= args.chain_name_mode <= 1):
        raise ValueError("Invalid value for chain_name_mode. It should be either 0 or 1.")

    # if not (0 <= args.write_mapping <= 1):
    #     raise ValueError("Invalid value for write_mapping. It should be either 0 or 1.")

    if not (1 <= args.coord_store_mode <= 2):
        raise ValueError("Invalid value for coord_store_mode. It should be between 1 and 2.")

    # if not (0 <= args.write_lookup <= 1):
    #     raise ValueError("Invalid value for write_lookup. It should be either 0 or 1.")

    # if not (0 <= args.db_output <= 1):
    #     raise ValueError("Invalid value for db_output. It should be either 0 or 1.")

    return args


def check_mash_arguments(args: str = "") -> Tuple[Optional[Namespace], Optional[Namespace]]:
    """
    Validate the custom arguments provided to DataSAIL for executing MASH.

    Args:
        args: String of the arguments that can be set by user. This should contain "|" as a separator between sketch
        and dist arguments.

    Returns:
        The namespace containing the parsed and validated arguments.
    """
    if os.path.isfile(args):
        raise NotImplementedError()
    else:
        arg_array = args.split("|")
        sketch_args, dist_args = None, None
        if len(arg_array) > 0:
            sketch_args = check_mash_sketch_arguments(arg_array[0])
        if len(arg_array) > 1:
            dist_args = check_mash_dist_arguments(arg_array[1])

    return sketch_args, dist_args


def check_mash_sketch_arguments(args: str = "") -> Namespace:
    """
    Validate the custom arguments provided to DataSAIL for executing MASH sketch.

    Args:
        args: String of the arguments that can be set by user

    Returns:
        The namespace containing the parsed and validated arguments.
    """
    # args = args.split(" ") if " " in args else (args if isinstance(args, list) else [args])
    args = MultiYAMLParser(MASH_SKETCH).parse_args(args)

    # Check if _p is valid
    if args.p < 1:
        raise ValueError("Invalid value for -p. It should be greater than or equal to 1.")

    # Check if -k is within the valid range
    if not (1 <= args.k <= 32):
        raise ValueError("Invalid value for -k. It should be between 1 and 32.")

    # Check if -s is within the valid range
    if args.s < 1:
        raise ValueError("Invalid value for -s. It should be greater than or equal to 1.")

    # Check if -S is within the valid range
    if not (0 <= args.S <= 4294967296):
        raise ValueError("Invalid value for -S. It should be between 0 and 4294967296.")

    # Check if -w is within the valid range
    if not (0 <= args.w <= 1):
        raise ValueError("Invalid value for -w. It should be between 0 and 1.")

    # Check if -b is a valid size
    if args.b != "":
        valid_suffixes = ['B', 'K', 'M', 'G', 'T']
        size_str = args.b.upper()
        suffix = size_str[-1]
        if suffix not in valid_suffixes:
            raise ValueError("Invalid suffix for -b. Use one of: B, K, M, G, T.")
        try:
            size = int(size_str[:-1])
            if size <= 0:
                raise ValueError("Invalid value for -b. Size must be greater than 0.")
        except ValueError:
            raise ValueError("Invalid value for -b. Numeric size is expected before the suffix.")

    # Check if -m is within the valid range
    if args.m <= 0:
        raise ValueError("Invalid value for -m. It should be greater than 0.")

    # Check if -c is within the valid range
    if args.c <= 0:
        raise ValueError("Invalid value for -c. It should be greater than 0.")

    # Check if -g is a valid size
    if args.g != "":
        valid_suffixes = ['B', 'K', 'M', 'G', 'T']
        size_str = args.g.upper()
        suffix = size_str[-1]
        if suffix not in valid_suffixes:
            raise ValueError("Invalid suffix for -g. Use one of: B, K, M, G, T.")
        try:
            size = int(size_str[:-1])
            if size <= 0:
                raise ValueError("Invalid value for -g. Size must be greater than 0.")
        except ValueError:
            raise ValueError("Invalid value for -g. Numeric size is expected before the suffix.")

    # Additional checks for conflicting options
    if args.a and args.z != "":
        raise ValueError("Options -a and -z are mutually exclusive.")

    if args.r and args.i:
        raise ValueError("Options -r and -i are mutually exclusive.")

    if args.b != "" and args.r:
        raise ValueError("Option -b implies -r.")

    if args.m != 1 and args.r:
        raise ValueError("Option -m implies -r.")

    if args.c != 1 and args.r:
        raise ValueError("Option -c implies -r.")

    if args.g != "" and args.r:
        raise ValueError("Option -g implies -r.")

    if args.a and args.k not in [9, 21]:
        raise ValueError("Option -a implies -k 9.")

    # if args.l and args.i:
    #     raise ValueError("Option -l and -i are mutually exclusive.")

    if args.n and (args.a or args.z != ""):
        raise ValueError("Option -n is implied by -a or -z.")

    if args.Z and (args.a or args.z != ""):
        raise ValueError("Option -Z is implied by -a or -z.")
    return args


def check_mash_dist_arguments(args: str = ""):
    """
    Validate the custom arguments provided to DataSAIL for executing MASH dist.

    Args:
        args: String of the arguments that can be set by user

    Returns:
        The namespace containing the parsed and validated arguments.
    """
    # args = args.split(" ") if " " in args else (args if isinstance(args, list) else [args])
    args = MultiYAMLParser(MASH_DIST).parse_args(args)

    if not (0 <= args.v <= 1):
        raise ValueError("Invalid value for -v. It should be between 0 and 1.")
    if not (0 <= args.d <= 1):
        raise ValueError("Invalid value for -d. It should be between 0 and 1.")
    if not (1 <= args.k <= 32):
        raise ValueError("Invalid value for -k. It should be between 1 and 32.")
    if args.s <= 0:
        raise ValueError("Invalid value for -s. It should be greater than 0.")
    if not (0 <= args.S <= 4294967296):
        raise ValueError("Invalid value for -S. It should be between 0 and 4294967296.")
    if not (0 <= args.w <= 1):
        raise ValueError("Invalid value for -w. It should be between 0 and 1.")
    if args.m <= 0:
        raise ValueError("Invalid value for -m. It should be greater than 0.")
    if args.c <= 0:
        raise ValueError("Invalid value for -c. It should be greater than 0.")

    if args.g != "":
        valid_suffixes = ['B', 'K', 'M', 'G', 'T']
        size_str = args.g.upper()
        suffix = size_str[-1]
        if suffix not in valid_suffixes:
            raise ValueError("Invalid suffix for -g. Use one of: B, K, M, G, T.")
        try:
            size = int(size_str[:-1])
            if size <= 0:
                raise ValueError("Invalid value for -g. Size must be greater than 0.")
        except ValueError:
            raise ValueError("Invalid value for -g. Numeric size is expected before the suffix.")

    if args.a and args.z != "":
        raise ValueError("Options -a and -z are mutually exclusive.")
    if args.r and args.i:
        raise ValueError("Options -r and -i are mutually exclusive.")
    if args.b != "" and args.r:
        raise ValueError("Option -b implies -r.")
    if args.c != 1 and args.r:
        raise ValueError("Option -c implies -r.")
    if args.g != "" and args.r:
        raise ValueError("Option -g implies -r.")
    if args.a and args.k not in [9, 21]:
        raise ValueError("Option -a implies -k 9.")
    # if args.l and args.i:
    #     raise ValueError("Option -l and -i are mutually exclusive.")
    if args.n and (args.a or args.z != ""):
        raise ValueError("Option -n is implied by -a or -z.")
    if args.Z and (args.a or args.z != ""):
        raise ValueError("Option -Z is implied by -a or -z.")
    return args
