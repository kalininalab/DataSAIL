from argparse import Namespace
from typing import Tuple, Union, Optional

from datasail.parsers import MultiYAMLParser
from datasail.settings import CDHIT, MMSEQS2, MASH, FOLDSEEK, MMSEQS, get_default, CDHIT_EST, MMSEQSPP, DIAMOND


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
    sim_on, dist_on = isinstance(similarity, str), isinstance(distance, str)
    both_none = not sim_on and not dist_on
    if (sim_on and similarity.lower().startswith(CDHIT_EST)) or \
            (both_none and get_default(dtype, dformat)[0] == CDHIT_EST):
        return check_cdhit_est_arguments(tool_args)
    elif (sim_on and similarity.lower().startswith(CDHIT)) or (both_none and get_default(dtype, dformat)[0] == CDHIT):
        return check_cdhit_arguments(tool_args)
    elif (sim_on and similarity.lower().startswith(DIAMOND)) or \
            (both_none and get_default(dtype, dformat)[0] == DIAMOND):
        return check_diamond_arguments(tool_args)
    elif (sim_on and similarity.lower().startswith(MMSEQSPP)) or \
            (both_none and get_default(dtype, dformat)[0] == MMSEQSPP):
        return check_mmseqspp_arguments(tool_args)
    elif (sim_on and similarity.lower().startswith(MMSEQS)) or \
            (both_none and get_default(dtype, dformat)[0] in [MMSEQS, MMSEQS2]):
        return check_mmseqs_arguments(tool_args)
    elif (sim_on and similarity.lower().startswith(FOLDSEEK)) or \
            (both_none and get_default(dtype, dformat)[0] == FOLDSEEK):
        return check_foldseek_arguments(tool_args)
    elif (dist_on and distance.lower().startswith(MASH)) or (both_none and get_default(dtype, dformat)[1] == MASH):
        return check_mash_arguments(tool_args)
    else:
        return None


def check_cdhit_est_arguments(args: str = "") -> Namespace:
    """
    Validate the custom arguments provided to DataSAIL for executing CD-HIT-EST.

    Args:
        args: String of the arguments that can be set by user
    """
    # args = args.split(" ") if " " in args else (args if isinstance(args, list) else [args])
    args = MultiYAMLParser(CDHIT_EST).parse_args(args)
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
    if args.g not in [0, 1]:
        raise ValueError("Invalid value for -g. It should be either 0 or 1.")
    # if not ((args.n == 5 and 0.8 <= args.c < 0.85) or
    #         (args.n == 6 and 0.85 <= args.c < 0.88) or
    #         (args.n == 7 and 0.88 <= args.c < 0.9) or
    #         (args.n in [8, 9, 10] and 0.9 <= args.c <= 1.0)):
    #     raise ValueError("There are restrictions on the values for n and c in CD-HIT:\n"
    #                      "n == 5 <=> c in [0.8, 0.85]\n"
    #                      "n == 6 <=> c in [0.85, 0.88]\n"
    #                      "n == 5 <=> c in [0.88, 0.9]\n"
    #                      "n in [8, 9, 10] <=> c in [0.9, 1.0]")

    # Check other values are within the valid range
    if not (1 <= args.b <= 32):
        raise ValueError("Invalid value for -b. It should be between 1 and 32.")
    if not (0 <= args.M):
        raise ValueError("Invalid value for -M. It should be greater than or equal to 0.")
    if not (0 <= args.n):
        raise ValueError("Invalid value for -n. It should be greater than or equal to 0.")
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


def check_diamond_arguments(args: str = "") -> Optional[Namespace]:
    """
    Validate the custom arguments provided to DataSAIL for executing DIAMOND.

    Args:
        args: String of the arguments that can be set by user

    Returns:
        The namespace containing the parsed and validated arguments
    """
    args = MultiYAMLParser(DIAMOND).parse_args(args)

    # Checking --comp-based-stats
    if not (0 <= args.comp_based_stats <= 4):
        raise ValueError("Invalid value for --comp-based-stats. It should be between 0 and 4.")

    # Checking --masking
    valid_masking_values = ["0", "none", "1", "seg", "tantan"]
    if args.masking not in valid_masking_values:
        raise ValueError(f"Invalid value for --masking. It should be one of {valid_masking_values}.")

    # Checking --soft-masking
    valid_soft_masking_values = ["none", "seg", "tantan"]
    if args.soft_masking not in valid_soft_masking_values:
        raise ValueError(f"Invalid value for --soft-masking. It should be one of {valid_soft_masking_values}.")

    # Checking --evalue
    if args.evalue < 0:
        raise ValueError("Invalid value for --evalue. It should be greater than or equal to 0.")

    # Checking --motif-masking
    if args.motif_masking not in [0, 1]:
        raise ValueError("Invalid value for --motif-masking. It should be 0 or 1.")

    # Checking --approx-id
    if not (0 <= args.approx_id <= 100):
        raise ValueError("Invalid value for --approx-id. It should be between 0 and 100.")

    # Checking --ext
    valid_ext_values = ["banded-fast", "banded-slow", "full"]
    if args.ext not in valid_ext_values:
        raise ValueError(f"Invalid value for --ext. It should be one of {valid_ext_values}.")

    # Checking --max-target-seqs
    if args.max_target_seqs < 0:
        raise ValueError("Invalid value for --max-target-seqs. It should be greater than or equal to 0.")

    # Checking --top
    if args.top is not None and (args.top <= 0 or args.top > 100):
        raise ValueError("Invalid value for --top. It should be between 0 and 100.")

    # Checking --shapes
    if args.shapes is not None and args.shapes < -1:
        raise ValueError("Invalid value for --shapes. It should be greater than 0.")

    # Checking --query
    if args.query is not None and not isinstance(args.query, str):
        raise ValueError("Invalid value for --query. It should be a string.")

    # Checking --strand
    valid_strand_values = ["both", "minus", "plus"]
    if args.strand not in valid_strand_values:
        raise ValueError(f"Invalid value for --strand. It should be one of {valid_strand_values}.")

    # Checking --unal
    if args.unal not in [0, 1]:
        raise ValueError("Invalid value for --unal. It should be 0 or 1.")

    # Checking --max-hsps
    if args.max_hsps < 0:
        raise ValueError("Invalid value for --max-hsps. It should be greater than or equal to 0.")

    # Checking --range-culling
    if args.range_culling not in [0, 1]:
        raise ValueError("Invalid value for --range-culling. It should be 0 or 1.")

    # Checking --compress
    if args.compress not in ["0", "1", "gzip", "zstd"]:
        raise ValueError("Invalid value for --compress. It should be 0 or 1.")

    # Checking --min-score
    if args.min_score is not None and args.min_score < 0:
        raise ValueError("Invalid value for --min-score. It should be greater than or equal to 0.")

    # Checking --id
    if args.id is not None and (args.id < 0 or args.id > 100):
        raise ValueError("Invalid value for --id. It should be between 0 and 100.")

    # Checking --query-cover
    if args.query_cover is not None and (args.query_cover < 0 or args.query_cover > 100):
        raise ValueError("Invalid value for --query-cover. It should be between 0 and 100.")

    # Checking --subject-cover
    if args.subject_cover is not None and (args.subject_cover < 0 or args.subject_cover > 100):
        raise ValueError("Invalid value for --subject-cover. It should be between 0 and 100.")

    # Checking --global-ranking
    if args.global_ranking is not None and args.global_ranking < -1:
        raise ValueError("Invalid value for --global-ranking. It should be greater than 0.")

    # Checking --block-size
    if args.block_size is not None and args.block_size <= 0:
        raise ValueError("Invalid value for --block-size. It should be greater than 0.")

    # Checking --index-chunks
    if args.index_chunks is not None and args.index_chunks <= 0:
        raise ValueError("Invalid value for --index-chunks. It should be greater than 0.")

    # Checking --gapopen
    if args.gapopen is not None and args.gapopen < 0:
        raise ValueError("Invalid value for --gapopen. It should be greater than or equal to 0.")

    # Checking --gapextend
    if args.gapextend is not None and args.gapextend < 0:
        raise ValueError("Invalid value for --gapextend. It should be greater than or equal to 0.")

    # Checking --matrix
    if args.matrix is not None and not args.matrix in ["BLOSUM45", "BLOSUM50", "BLOSUM62", "BLOSUM80", "BLOSUM90",
                                                       "PAM30", "PAM70", "PAM250"]:
        raise ValueError("Invalid value for --matrix. It must be one of the BLOSUM or PAM matrices.")

    # Checking --custom-matrix
    if args.custom_matrix is not None and not isinstance(args.custom_matrix, str):
        raise ValueError("Invalid value for --custom-matrix. It should be a string.")

    # Checking --frameshift
    valid_frameshift_values = ["disabled"]
    if args.frameshift not in valid_frameshift_values:
        raise ValueError(f"Invalid value for --frameshift. It should be one of {valid_frameshift_values}.")

    # Checking --long-reads
    if args.long_reads not in [0, 1]:
        raise ValueError("Invalid value for --long-reads. It should be 0 or 1.")

    # Checking --no-self-hits
    if args.no_self_hits not in [0, 1]:
        raise ValueError("Invalid value for --no-self-hits. It should be 0 or 1.")

    # Checking --skip-missing-seqids
    if args.skip_missing_seqids not in [0, 1]:
        raise ValueError("Invalid value for --skip-missing-seqids. It should be 0 or 1.")

    # Checking --file-buffer-size
    if args.file_buffer_size is not None and args.file_buffer_size <= 0:
        raise ValueError("Invalid value for --file-buffer-size. It should be greater than 0.")

    # Checking --bin
    if args.bin is not None and args.bin <= 0:
        raise ValueError("Invalid value for --bin. It must be positive.")

    # Checking --ext-chunk-size
    if args.ext_chunk_size is not None and args.ext_chunk_size != "auto" and int(args.ext_chunk_size) <= 0:
        raise ValueError("Invalid value for --ext-chunk-size. It should be greater than 0.")

    # Checking --dbsize
    if args.dbsize is not None and not (args.dbsize[:-1].isdigit() and args.dbsize[-1].lower() not in ["g", "m", "k"]):
        raise ValueError("Invalid value for --dbsize. It should be a positive integer.")

    # Checking --tantan-minMaskProb
    if args.tantan_minMaskProb is not None and (args.tantan_minMaskProb < 0 or args.tantan_minMaskProb > 1):
        raise ValueError("Invalid value for --tantan-minMaskProb. It should be between 0 and 1.")

    # Checking --algo
    valid_algo_values = ["0", "double-indexed", "1", "query-indexed", "ctg"]
    if args.algo not in valid_algo_values:
        raise ValueError(f"Invalid value for --algo. It should be one of {valid_algo_values}.")

    # Checking --min-orf
    if args.min_orf is not None and args.min_orf <= 0:
        raise ValueError("Invalid value for --min-orf. It should be a positive integer.")

    # Checking --seed-cut
    if args.seed_cut is not None and args.seed_cut < 0:
        raise ValueError("Invalid value for --seed-cut. It should be a non-negative value.")

    # Checking --freq-sd
    if args.freq_sd is not None and args.freq_sd < 0:
        raise ValueError("Invalid value for --freq-sd. It should be a non-negative value.")

    # Checking --id2
    if args.id2 is not None and args.id2 < 0:
        raise ValueError("Invalid value for --id2. It should be a non-negative integer.")

    # Checking --gapped-filter-evalue
    if args.gapped_filter_evalue is not None and args.gapped_filter_evalue != "auto" and args.gapped_filter_evalue < 0:
        raise ValueError("Invalid value for --gapped-filter-evalue. It should be greater than or equal to 0 or 'auto'.")

    # Checking --band
    # if args.band is not None and args.band <= 0:
    #     raise ValueError("Invalid value for --band. It should be greater than 0.")

    # Checking --shape-mask
    # if args.shape_mask is not None and not isinstance(args.shape_mask, str):
    #     raise ValueError("Invalid value for --shape-mask. It should be a string.")

    # Checking --culling-overlap
    if args.culling_overlap is not None and (int(args.culling_overlap[:-1]) < 0 or
                                             int(args.culling_overlap[:-1]) > 100 or
                                             args.culling_overlap[-1] != "%"):
        raise ValueError("Invalid value for --culling-overlap. It should be between 0% and 100%.")

    # Checking --taxon-k
    if args.taxon_k is not None and args.taxon_k < -1:
        raise ValueError("Invalid value for --taxon-k. It should be a non-negative integer or -1.")

    # Checking --range-cover
    if args.range_cover is not None and (int(args.range_cover[:-1]) < 0 or int(args.range_cover[:-1]) > 100 or
                                         args.range_cover[-1] != "%"):
        raise ValueError("Invalid value for --range-cover. It should be between 0% and 100%.")

    # Checking --stop-match-score
    if args.stop_match_score is not None and not (args.stop_match_score == -1 or args.stop_match_score > 0):
        raise ValueError("Invalid value for --stop-match-score. Is should be a non-negative integer or -1.")

    # Checking --window
    if args.window is not None and args.window <= 0:
        raise ValueError("Invalid value for --window. It should be greater than 0.")

    # Checking --ungapped-score
    if args.ungapped_score is not None and args.ungapped_score < 0:
        raise ValueError("Invalid value for --ungapped-score. It should be greater than or equal to 0.")

    # Checking --hit-band
    # if args.hit_band is not None and args.hit_band <= 0:
    #     raise ValueError("Invalid value for --hit-band. It should be greater than 0.")

    # Checking --hit-score
    if args.hit_score is not None and args.hit_score < 0:
        raise ValueError("Invalid value for --hit-score. It should be greater than or equal to 0.")

    # Checking --gapped-xdrop
    # if args.gapped_xdrop is not None and args.gapped_xdrop <= 0:
    #     raise ValueError("Invalid value for --gapped-xdrop. It should be greater than 0.")

    # Checking --rank-ratio2
    if args.rank_ratio2 is not None and args.rank_ratio2 <= 0:
        raise ValueError("Invalid value for --rank-ratio2. It should be greater than 0.")

    # Checking --rank-ratio
    if args.rank_ratio is not None and args.rank_ratio <= 0:
        raise ValueError("Invalid value for --rank-ratio. It should be greater than 0.")

    # Checking --lambda
    # if getattr(args, "lambda") is not None and getattr(args, "lambda") <= 0:
    #     raise ValueError("Invalid value for --lambda. It should be greater than 0.")

    # Checking --K
    # if args.K is not None and args.K <= 0:
    #     raise ValueError("Invalid value for --K. It should be greater than 0.")

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
    for arg_name in ["k", "max_seqs", "split", "zdrop", "id_offset",
                     "cluster_steps", "max_rejected", "max_accept", "realign_max_seqs", "min_aln_len", "hash_shift",
                     "kmer_per_seq"]:
        if not (0 <= getattr(args, arg_name) <= 2147483647):
            raise ValueError(f"Invalid value for --{arg_name.replace('_', '-')}. "
                             f"It should be a non-negative integer.")

    # Check floating-point values  # Add: "realign_score_bias" ?
    for arg_name in ["score_bias", "corr_score_weight", "corr_score_weight"]:
        if getattr(args, arg_name) < 0.0:
            raise ValueError(f"Invalid value for --{arg_name.replace('_', '-')}. "
                             f"It should be a non-negative float.")

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


def check_mmseqspp_arguments(args: str = "") -> Optional[Namespace]:
    """
    Validate the custom arguments provided to DataSAIL for executing MMSEQS++. That is mmseqs used for computing
    similarity matrix between the sequences.

    Args:
        args: String of the arguments that can be set by user

    Returns:
        The namespace containing the parsed and validated arguments.
    """
    args = MultiYAMLParser(MMSEQSPP).parse_args(args)

    # Check specific conditions for certain arguments
    if not 1.0 <= args.s <= 7.5:
        raise ValueError("Error: Sensitivity must be between 1.0 and 7.5.")

    # if args.gap_open < 0 or (args.alph_size == 'aa' and args.gap_open > 11):
    #     raise ValueError("Error: Gap open cost is out of valid range.")

    # if args.gap_extend < 0 or (args.alph_size == 'aa' and args.gap_extend > 11):
    #     raise ValueError("Error: Gap extension cost is out of valid range.")

    if args.k < 0:
        raise ValueError("Error: k-mer length must be a non-negative integer.")

    if not 0.0 <= args.mask_prob <= 1.0:
        raise ValueError("Error: Mask probability must be between 0 and 1.")

    if args.alignment_mode not in [0, 1, 2, 3]:
        raise ValueError("Error: Alignment mode must be 0, 1, 2, or 3.")

    if args.alignment_output_mode not in [0, 1, 2, 3, 4, 5]:
        raise ValueError("Error: Alignment output mode must be 0, 1, 2, 3, 4, or 5.")

    if args.alt_ali < 0:
        raise ValueError("Error: Show up to this many alternative alignments must be a non-negative integer.")

    if not 0.0 <= args.corr_score_weight:
        raise ValueError("Error: Weight of backtrace correlation score must be a non-negative float.")

    if args.diag_score not in [0, 1]:
        raise ValueError("Error: Ungapped diagonal scoring during prefilter must be 0 or 1.")

    if args.exact_kmer_matching not in [0, 1]:
        raise ValueError("Error: Extract only exact k-mers for matching must be 0 or 1.")

    if args.k < 0:
        raise ValueError("Error: k-mer length must be a non-negative integer.")

    if args.mask not in [0, 1]:
        raise ValueError("Error: Mask argument must be 0 or 1.")

    if not 0.0 <= args.mask_prob <= 1.0:
        raise ValueError("Error: Mask probability must be between 0.0 and 1.0.")

    if args.mask_lower_case not in [0, 1]:
        raise ValueError("Error: Mask lower case argument must be 0 or 1.")

    if args.max_accept < 0:
        raise ValueError("Error: Maximum accepted alignments must be a non-negative integer.")

    if args.max_rejected < 0:
        raise ValueError("Error: Maximum rejected alignments must be a non-negative integer.")

    if args.max_seqs < 1:
        raise ValueError("Error: Maximum results per query sequence must be at least 1.")

    if args.min_aln_len < 0:
        raise ValueError("Error: Minimum alignment length must be a non-negative integer.")

    if args.min_ungapped_score < 0:
        raise ValueError("Error: Minimum ungapped alignment score must be a non-negative integer.")

    if args.realign_score_bias < -1.0 or args.realign_score_bias > 1.0:
        raise ValueError("Error: Realign score bias must be between -1.0 and 1.0.")

    if args.realign_max_seqs < 1:
        raise ValueError("Error: Maximum number of results for realignment must be at least 1.")

    if args.spaced_kmer_mode not in [0, 1]:
        raise ValueError("Error: Spaced k-mer mode must be 0 or 1.")

    if args.split_mode not in [0, 1, 2]:
        raise ValueError("Error: Split mode must be 0, 1, or 2.")

    if args.cov_mode not in [0, 1, 2, 3, 4, 5]:
        raise ValueError("Error: Coverage mode must be 0, 1, 2, 3, 4, or 5.")

    if args.db_load_mode not in [0, 1, 2, 3]:
        raise ValueError("Error: Database preload mode must be 0, 1, 2, or 3.")

    if args.seq_id_mode not in [0, 1, 2]:
        raise ValueError("Error: Sequence identity mode must be 0, 1, or 2.")

    if args.pca < 0:
        raise ValueError("Error: PCA must be a non-negative integer.")

    if args.pcb < 0:
        raise ValueError("Error: PCB must be a non-negative integer.")

    if args.score_bias < 0:
        raise ValueError("Error: The score bias must be a non-negative integer.")

    if args.split < 0:
        raise ValueError("Error: The number of splits must be a non-negative integer.")

    if args.zdrop < 0:
        raise ValueError("Error: zdrop must be a non-negative integer.")

    if not (0 <= args.max_seq_len <= 65536):
        raise ValueError("Invalid value for max_seq_len. It should be between 0 and 65536.")

    if args.comp_bias_corr not in [0, 1]:
        raise ValueError("Error: Composition bias correction must be 0 or 1.")

    if not (0.0 <= args.comp_bias_corr_scale <= 1.0):
        raise ValueError("Error: Composition bias correction scale must be between 0.0 and 1.0.")

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

    if not (0 <= args.min_aln_len <= 2147483647):
        raise ValueError("Invalid value for min_aln_len. It should be between 0 and 2147483647.")

    if not (0 <= args.seq_id_mode <= 2):
        raise ValueError("Invalid value for seq_id_mode. It should be between 0 and 2.")

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

    if not (0 <= args.greedy_best_hits <= 1):
        raise ValueError("Invalid value for greedy_best_hits. It should be either 0 or 1.")

    if not (0 <= args.db_load_mode <= 3):
        raise ValueError("Invalid value for db_load_mode. It should be between 0 and 3.")

    if not (0 <= args.max_seq_len <= 65536):
        raise ValueError("Invalid value for max_seq_len. It should be between 0 and 65536.")

    if not (0 <= args.zdrop <= 2147483647):
        raise ValueError("Invalid value for zdrop. It should be between 0 and 2147483647.")

    if not (0 <= args.chain_name_mode <= 1):
        raise ValueError("Invalid value for chain_name_mode. It should be either 0 or 1.")

    if not (1 <= args.coord_store_mode <= 2):
        raise ValueError("Invalid value for coord_store_mode. It should be between 1 and 2.")

    return args


def check_mash_arguments(args: str = "") -> Namespace:
    args = MultiYAMLParser(MASH).parse_args(args)

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
    if args.m != 1 and args.r:
        raise ValueError("Option -m implies -r.")
    if args.c != 1 and args.r:
        raise ValueError("Option -c implies -r.")
    if args.g != "" and args.r:
        raise ValueError("Option -g implies -r.")
    if args.a and args.k not in [9, 21]:
        raise ValueError("Option -a implies -k 9.")
    if args.n and (args.a or args.z != ""):
        raise ValueError("Option -n is implied by -a or -z.")
    if args.Z and (args.a or args.z != ""):
        raise ValueError("Option -Z is implied by -a or -z.")
    return args
