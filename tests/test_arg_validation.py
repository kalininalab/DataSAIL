import pytest

from datasail.validation.validate import check_cdhit_arguments, check_foldseek_arguments, check_mmseqs_arguments, \
    check_mmseqspp_arguments, check_mash_arguments, check_diamond_arguments
from datasail.sail import datasail


def test_validate_args():
    # Test that validate_args raises an error when splits are less than 2
    with pytest.raises(ValueError):
        datasail(splits=[0.5])

    # Test that validate_args raises an error when splits and names have different lengths
    with pytest.raises(ValueError):
        datasail(splits=[0.5, 0.5], names=['Split1'])

    # Test that validate_args raises an error when splits have the same name
    with pytest.raises(ValueError):
        datasail(splits=[0.5, 0.5], names=['Split1', 'Split1'])

    # Test that validate_args raises an error when max_sec is less than 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], max_sec=0)

    # Test that validate_args raises an error when threads is less than 0
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], threads=-1)

    # Test that validate_args raises an error when delta is not between 0 and 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], delta=1.5)

    # Test that validate_args raises an error when epsilon is not between 0 and 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], epsilon=1.5)

    # Test that validate_args raises an error when runs is less than 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], runs=0)

    # Test that validate_args raises an error when linkage is not one of 'average', 'single', or 'complete'
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], linkage='invalid')

    # Test that validate_args raises an error when e_clusters is less than 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], e_clusters=0)

    # Test that validate_args raises an error when f_clusters is less than 1
    with pytest.raises(ValueError):
        datasail(splits=[0.7, 0.3], f_clusters=0)


@pytest.mark.parametrize("args", [
    "-c -1", "-c 2", "-s -0.5", "-s 2", "-aL -1", "-aL 2", "-aS -0.1", "-aS 2", "-uL -1", "-uL 2", "-uS -0.1", "-uS 2",
    "-G 2", "-g 2", "-b 0", "-b 33", "-M -1", "-n -1", "-t -1", "-S -1", "-S 4294967297", "-AL -1", "-AS -1", "-A -1",
    "-U -1",
])
def test_cdhit_parser_invalid(args):
    with pytest.raises(ValueError):
        check_cdhit_arguments(args)


@pytest.mark.parametrize("args", [
    "-c 0.4 -n 2", "-c 1.0", "-s 0.0", "-s 1.0", "-aL 0.0", "-aL 1.0", "-aS 0.0", "-aS 1.0", "-uL 0.0", "-uL 1.0",
    "-uS 0.0", "-uS 1.0", "-G 0", "-G 1", "-g 0", "-g 1", "-b 1", "-b 32", "-M 0", "-c 0.9 -n 5", "-t 0", "-S 0",
    "-S 4294967296", "-AL 0", "-AL 99999999", "-AS 0", "-AS 99999999", "-A 0", "-A 99999999", "-U 0", "-U 99999999",
])
def test_cdhit_parser_valid(args):
    assert check_cdhit_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "--comp-based-stats 5", "--masking unknown", "--soft-masking invalid", "--evalue -0.001", "--motif-masking 2",
    "--approx-id 110", "--ext invalid", "--max-target-seqs -5", "--top 110", "--shapes -3",
    "--strand neither", "--unal 2", "--max-hsps -2", "--compress 2", "--min-score -10", "--id 110", "--query-cover 120",
    "--subject-cover 110", "--global-ranking -50", "--block-size -2.0",
    "--index-chunks -4", "--gapopen -15", "--gapextend -25", "--matrix unknown_matrix", "--frameshift enabled",
    "--file-buffer-size -67108864", "--bin -1", "--ext-chunk-size -10", "--dbsize invalid_size",
    "--tantan-minMaskProb 1.5", "--algo unknown-algorithm", "--min-orf -30", "--seed-cut -5", "--freq-sd -3",
    "--gapped-filter-evalue -0.1", "--culling-overlap -10", "--taxon-k -5", "--range-cover 110",
    "--stop-match-score -0.5", "--window 0", "--ungapped-score -5", "--rank-ratio2 -0.8", "--rank-ratio -0.9"
])
def test_diamond_args_checker_invalid(args):
    with pytest.raises(ValueError):
        check_diamond_arguments(args)


@pytest.mark.parametrize("args", [
    "--comp-based-stats 2", "--masking seg", "--soft-masking tantan", "--evalue 0.001", "--motif-masking 1",
    "--approx-id 80", "--ext full", "--max-target-seqs 25", "--top 10", "--shapes 5", "--query input.fasta",
    "--strand both", "--unal 1", "--max-hsps 1", # "--range-culling \"1%\"",
    "--compress 1", "--min-score 50", "--id 90", "--query-cover 80", "--subject-cover 70", "--global-ranking 100",
    "--block-size 2.0", "--index-chunks 4", "--parallel-tmpdir /tmp", # "--gapopen \"-1\"", "--gapextend \"-2\"",
    "--matrix BLOSUM62", "--custom-matrix custom_matrix.txt", "--frameshift disabled", "--file-buffer-size 67108864", "--bin 10",
    "--ext-chunk-size auto", "--dbsize 1000000000", "--tantan-minMaskProb 0.9", "--algo double-indexed", "--min-orf 50",
    "--seed-cut 10", "--freq-masking 1", "--freq-sd 3", "--id2 50", "--gapped-filter-evalue 0.5", "--band 50",
    "--shape-mask AACGT,GGTCA", "--taxon-k 5", # "--culling-overlap 50%", "--range-cover 50%",
    "--stop-match-score 0.5", "--window 5", "--ungapped-score 30", "--hit-band 10", "--hit-score 50", "--gapped-xdrop 20",
    "--rank-ratio2 0.8", "--rank-ratio 0.9", "--lambda 0.5", "--K 10"
])
def test_diamond_args_checker_valid(args):
    assert check_diamond_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-s 1.0", "--comp-bias-corr 0", "--exact-kmer-matching 0", "--mask-prob 0", "--min-ungapped-score 0", "-c 0",
    "-e 0", "--min-seq-id 0", "--seq-id-mode 0", "--split-mode 0", "--cov-mode 0", "--alignment-mode 0",
    "--cluster-mode 0", "--similarity-type 1", "--rescore-mode 0", "--dbtype 0", "--createdb-mode 0",
    "--max-seq-len 1", "--max-iterations 1", "--min-aln-len 0", "--mask 0", "--mask-lower-case 0",
    "--spaced-kmer-mode 0", "--sort-results 0", "-k 1", "--max-seqs 1", "--split 1", "--zdrop 1",
    "--id-offset 0", "--cluster-steps 0", "--max-rejected 1", "--max-accept 1",
    "--realign-max-seqs 1", "--min-aln-len 1", "--hash-shift 0", "--kmer-per-seq 1", "--score-bias 0.0",
    "--corr-score-weight 0.0", "--corr-score-weight 0.0", "--k-score seq:0.0,prof:0.0", "--alph-size aa:2,nucl:21",
    "--gap-open aa:0.0,nucl:0.0", "--gap-extend aa:0.0,nucl:0.0", "--sub-mat aa:0.0,nucl:0.0",
    "--kmer-per-seq-scale aa:0.0,nucl:0.0",
])
def test_mmseqs_parser_valid(args):
    assert check_mmseqs_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-s 0", "--comp-bias-corr -1", "--exact-kmer-matching -1", "--mask-prob -1", "--min-ungapped-score -1", "-c -1",
    "-e -1", "--min-seq-id -1", "--seq-id-mode -1", "--split-mode -1", "--cov-mode -1", "--alignment-mode -1",
    "--cluster-mode -1", "--similarity-type -1", "--rescore-mode -1", "--dbtype -1", "--createdb-mode -1",
    "--max-seq-len 0", "--max-iterations 0", "--min-seq-id -1", "--mask 2", "--mask-lower-case 2",
    "--spaced-kmer-mode 2", "--sort-results 2", "-k -1", "--max-seqs -1", "--split -1", "--zdrop -1",
    "--id-offset -1", "--cluster-steps -1", "--max-rejected -1", "--max-accept -1",
    "--realign-max-seqs -1", "--min-aln-len -1", "--hash-shift -1", "--kmer-per-seq -1",
    "--score-bias -1.0", "--corr-score-weight -1.0", "--corr-score-weight -1.0", "--k-score invalid-format",
    "--alph-size invalid-format", "--gap-open invalid-format", "--gap-extend invalid-format",
    "--sub-mat invalid-format", "--kmer-per-seq-scale invalid-format",
])
def test_mmseqs_parser_invalid(args):
    with pytest.raises(ValueError):
        check_mmseqs_arguments(args)


@pytest.mark.parametrize("args", [
    "-s 4.0", "-k 15", "--mask-prob 0.8", "--mask-lower-case 1",
    "--max-accept 100", "--max-rejected 50", "--max-seqs 500", "--min-aln-len 20",
    "--min-ungapped-score 20", "--realign", "--realign-score-bias 0.1", "--realign-max-seqs 10",
    "--spaced-kmer-mode 1", "--split-mode 2", "--split-memory-limit 2G", "--cov-mode 3",
    "--pca 5", "--pcb 10", "--sub-mat [aa:blosum80.out,nucl:nucleotide.out]",
    "--max-seq-len 5000", "--db-load-mode 1", "--alignment-mode 2", "--alignment-output-mode 4",
    "--alph-size [aa:20,nucl:4]", "--alt-ali 3", "--corr-score-weight 0.005", "--diag-score",
    "--exact-kmer-matching 1", "--mask 0", "--mask-prob 0.95", "--mask-lower-case 0",
    "--max-accept 200", "--max-rejected 100", "--max-seqs 1000", "--min-aln-len 30",
    "--min-ungapped-score 30", "--realign-score-bias -0.1", "--realign-max-seqs 20",
    "-s 7.5", "--score-bias 0.2", "--seed-sub-mat [aa:VTML80.out,nucl:nucleotide.out]",
    "--seq-id-mode 1", "--spaced-kmer-mode 0", "--spaced-kmer-pattern '1-2-1'",
    "--split 4", "--split-mode 1", "--split-memory-limit 1G", "--taxon-list '123,456,789'",
    "--wrapped-scoring", "--zdrop 50", "--add-self-matches", "--comp-bias-corr 0",
    "--comp-bias-corr-scale 0.8", "--cov-mode 5", "--pca 2", "--pcb 5", "--add-self-matches",
    "--sub-mat [aa:blosum45.out,nucl:nucleotide.out]", "--max-seq-len 10000", "--db-load-mode 3",
])
def test_mmseqspp_parser_valid(args):
    assert check_mmseqspp_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-s -0.5", "-k -1", "--mask-prob 1.5", "--mask-lower-case 2",
    "--max-accept -100", "--max-rejected -50", "--max-seqs 0", "--min-aln-len -20",
    "--min-ungapped-score -20", "--realign-score-bias -1.5", "--realign-max-seqs 0",
    "--spaced-kmer-mode 2", "--split-mode 3", "--cov-mode 6",
    "--pca -5", "--pcb -10", "--max-seq-len -5000", "--db-load-mode 4",
    "--alignment-mode 4", "--alignment-output-mode 6",
    "--alt-ali -3", "--corr-score-weight -0.005", "--exact-kmer-matching 2",
    "--mask 2", "--mask-prob 1.05", "--mask-lower-case 2", "--max-accept -200", "--max-rejected -100",
    "--max-seqs 0", "--min-aln-len -30", "--min-ungapped-score -30",
    "--realign-score-bias 1.5", "--realign-max-seqs 0", "-s 8.0", "--score-bias -0.2",
    "--seq-id-mode 3", "--spaced-kmer-mode 2",
    "--split -4", "--split-mode 3", "--zdrop -50",
    "--comp-bias-corr 2", "--comp-bias-corr-scale -0.8", "--cov-mode 6", "--pca -2", "--pcb -5",
    "--max-seq-len -10000", "--db-load-mode 4",
])
def test_mmseqspp_parser_invalid(args):
    with pytest.raises(ValueError):
        check_mmseqspp_arguments(args)


@pytest.mark.parametrize("args", [
    "--comp-bias-corr 1", "--comp-bias-corr-scale 0.8", "-s 5.0", "-k 50", "--max-seqs 1000", "--split 200",
    "--split-mode 1", "--diag-score", "--exact-kmer-matching 1", "--mask 0", "--mask-prob 0.2",
    "--mask-lower-case 1", "--min-ungapped-score 500", "--spaced-kmer-mode 0", "--alignment-mode 2",
    "--alignment-output-mode 4", "--min-aln-len 100", "--seq-id-mode 1",
    "--num-iterations 10", "--tmscore-threshold 0.9", "--tmalign-hit-order 3", "--tmalign-fast 1",
    "--lddt-threshold 0.5", "--prefilter-mode 2", "--alignment-type 1", "--cluster-search 0",
    "--mask-bfactor-threshold 50", "--greedy-best-hits", "--db-load-mode 2", "--threads 8",
    "--max-seq-len 32768", "--zdrop 1000",
    "--chain-name-mode 0", "--coord-store-mode 2",
])
def test_foldseek_parser_valid(args):
    assert check_foldseek_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "--comp-bias-corr -1", "--comp-bias-corr 2", "--comp-bias-corr-scale -0.1", "--comp-bias-corr-scale 1.2",
    "-s 0.9", "-s 8.0", "-k -10", "-k 100", "--max-seqs -1", "--max-seqs 2147483648", "--split -1",
    "--split 2147483648", "--split-mode -1", "--split-mode 3",
    "--exact-kmer-matching -1", "--exact-kmer-matching 2", "--mask -1", "--mask 2", "--mask-prob -0.1",
    "--mask-prob 1.1", "--mask-lower-case -1", "--mask-lower-case 2", "--min-ungapped-score -1",
    "--min-ungapped-score 2147483648", "--spaced-kmer-mode -1", "--spaced-kmer-mode 2", "--alignment-mode -1",
    "--alignment-mode 4", "--alignment-output-mode -1", "--alignment-output-mode 6", "--min-aln-len -1",
    "--min-aln-len 2147483648", "--seq-id-mode -1", "--seq-id-mode 3",
    "--num-iterations -1", "--num-iterations 2147483648", "--tmscore-threshold -0.1", "--tmscore-threshold 1.1",
    "--tmalign-hit-order -1", "--tmalign-hit-order 5", "--tmalign-fast -1", "--tmalign-fast 2",
    "--lddt-threshold -0.1", "--lddt-threshold 1.1", "--prefilter-mode -1", "--prefilter-mode 3",
    "--alignment-type -1", "--alignment-type 3", "--cluster-search -1", "--cluster-search 2",
    "--mask-bfactor-threshold -1", "--mask-bfactor-threshold 101",
    "--db-load-mode -1", "--db-load-mode 4", "--max-seq-len -1", "--max-seq-len 65537",
    "--zdrop -1", "--zdrop 2147483648", "--chain-name-mode -1", "--chain-name-mode 2", "--coord-store-mode 0", "--coord-store-mode 3",
])
def test_foldseek_parser_invalid(args):
    with pytest.raises(ValueError):
        check_foldseek_arguments(args)


@pytest.mark.parametrize("args", [
    "-v 0.0", "-v 1.0", "-d 0.0", "-d 1.0", "-k 1", "-k 32", "-s 1", "-S 0", "-S 4294967296", "-w 0", "-w 1",
    "-b 1B", "-b 2K", "-m 10", "-c 0.1", "-g 1B", "-g 2K", "-a -k 9",
])
def test_mash_parser_valid(args):
    assert check_mash_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-v -0.1", "-v 1.1", "-d -0.1", "-d 1.1", "-k 0", "-k 33", "-s 0", "-S -1", "-S 4294967297", "-w -0.1",
    "-w 1.1", "-m 0", "-c 0", "-g 0B", "-g ABC", "-a -z -a", "-r -i -r", "-c 0", "-g B",
    "-a -k 8", "-n -a", "-Z -z",
])
def test_mash_parser_invalid(args):
    with pytest.raises(ValueError):
        check_mash_arguments(args)


def test_check_booleans():
    args = check_foldseek_arguments()
    assert args.diag_score  # Example for default positive value
    assert not args.exhaustive_search  # Example for default negative value
