import pytest

from datasail.reader.validate import check_cdhit_arguments, check_foldseek_arguments, check_mmseqs_arguments, \
    check_mash_dist_arguments, check_mash_sketch_arguments, check_mash_arguments


@pytest.mark.parametrize("args", [
    "-c -1", "-c 2", "-s -0.5", "-s 2", "-aL -1", "-aL 2", "-aS -0.1", "-aS 2", "-uL -1", "-uL 2", "-uS -0.1", "-uS 2",
    "-G 2", "-g 2", "-b 0", "-b 33", "-M -1", "-T -1", "-n -1", "-t -1",
    "-S -1", "-S 4294967297", "-AL -1", "-AS -1", "-A -1", "-U -1",
])
def test_cdhit_parser_invalid(args):
    with pytest.raises(ValueError):
        check_cdhit_arguments(args)


@pytest.mark.parametrize("args", [
    "-c 0.4 -n 2", "-c 1.0", "-s 0.0", "-s 1.0", "-aL 0.0", "-aL 1.0", "-aS 0.0", "-aS 1.0", "-uL 0.0", "-uL 1.0",
    "-uS 0.0", "-uS 1.0", "-G 0", "-G 1", "-g 0", "-g 1", "-b 1", "-b 32", "-M 0", "-T 0", "-c 0.9 -n 5", "-t 0", "-S 0", "-S 4294967296", "-AL 0",
    "-AL 99999999", "-AS 0", "-AS 99999999", "-A 0", "-A 99999999", "-U 0", "-U 99999999",
])
def test_cdhit_parser_valid(args):
    assert check_cdhit_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-s 1.0", "--comp-bias-corr 0", "--exact-kmer-matching 0", "--mask-prob 0", "--min-ungapped-score 0", "-c 0",
    "-e 0", "--min-seq-id 0", "--seq-id-mode 0", "--split-mode 0", "--cov-mode 0", "--alignment-mode 0",
    "--cluster-mode 0", "--similarity-type 1", "--rescore-mode 0", "--dbtype 0", "--createdb-mode 0",
    "--max-seq-len 1", "--max-iterations 1", "--min-aln-len 0", "--mask 0", "--mask-lower-case 0",
    "--spaced-kmer-mode 0", "--sort-results 0", "-k 1", "--max-seqs 1", "--split 1", "--threads 1", "--zdrop 1",
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
    "--spaced-kmer-mode 2", "--sort-results 2", "-k -1", "--max-seqs -1", "--split -1", "--threads -1", "--zdrop -1",
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
    "--comp-bias-corr 1", "--comp-bias-corr-scale 0.8", "-s 5.0", "-k 50", "--max-seqs 1000", "--split 200",
    "--split-mode 1", "--diag-score", "--exact-kmer-matching 1", "--mask 0", "--mask-prob 0.2",
    "--mask-lower-case 1", "--min-ungapped-score 500", "--spaced-kmer-mode 0", "--alignment-mode 2",
    "--alignment-output-mode 4", "-e 0.01", "--min-aln-len 100", "--seq-id-mode 1",
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
    "--alignment-mode 4", "--alignment-output-mode -1", "--alignment-output-mode 6", "-e -0.1", "--min-aln-len -1",
    "--min-aln-len 2147483648", "--seq-id-mode -1", "--seq-id-mode 3",
    "--num-iterations -1", "--num-iterations 2147483648", "--tmscore-threshold -0.1", "--tmscore-threshold 1.1",
    "--tmalign-hit-order -1", "--tmalign-hit-order 5", "--tmalign-fast -1", "--tmalign-fast 2",
    "--lddt-threshold -0.1", "--lddt-threshold 1.1", "--prefilter-mode -1", "--prefilter-mode 3",
    "--alignment-type -1", "--alignment-type 3", "--cluster-search -1", "--cluster-search 2",
    "--mask-bfactor-threshold -1", "--mask-bfactor-threshold 101",
    "--db-load-mode -1", "--db-load-mode 4", "--threads 0",
    "--threads 2147483648", "--max-seq-len -1", "--max-seq-len 65537",
    "--zdrop -1", "--zdrop 2147483648", "--chain-name-mode -1", "--chain-name-mode 2", "--coord-store-mode 0", "--coord-store-mode 3",
])
def test_foldseek_parser_invalid(args):
    with pytest.raises(ValueError):
        check_foldseek_arguments(args)


@pytest.mark.parametrize("args", [
    "-p 1", "-k 1", "-k 32", "-s 1", "-S 0", "-S 4294967296", "-w 0", "-w 1", "-b 1B", "-b 2K", "-m 10", "-c 0.1",
    "-g 1B", "-g 2K", "-a -k 9",
])
def test_mash_sketch_parser_valid(args):
    assert check_mash_sketch_arguments(args) is not None


@pytest.mark.parametrize("args", [
    "-p 0", "-k 0", "-k 33", "-s 0", "-S -1", "-S 4294967297", "-w -0.1", "-w 1.1", "-b 0B", "-b ABC", "-m 0", "-c 0",
    "-g 0B", "-g ABC", "-a -z -a", "-r -i -r", "-b 2", "-c 0", "-g B", "-a -k 8", "-n -a", "-Z -z",
])
def test_mash_sketch_parser_invalid(args):
    with pytest.raises(ValueError):
        check_mash_sketch_arguments(args)


@pytest.mark.parametrize("args", [
    "-v -0.1", "-v 1.1", "-d -0.1", "-d 1.1", "-k 0", "-k 33", "-s 0", "-S -1", "-S 4294967297", "-w -0.1", "-w 1.1",
    "-m 0", "-c 0", "-g 0B", "-g ABC", "-a -z", "-r -i", "-c 0", "-g B", "-a -k 8"
])
def test_mash_dist_parser_invalid(args):
    with pytest.raises(ValueError):
        check_mash_dist_arguments(args)


@pytest.mark.parametrize("args", [
    "-v 0.0", "-v 1.0", "-d 0.0", "-d 1.0", "-k 1", "-k 32", "-s 1", "-S 0", "-S 4294967296", "-w 0.0", "-w 1.0",
    "-m 10", "-c 1", "-g 1B", "-g 2K", "-a", "-z", "-r", "-i", "-b 2", "-n", "-Z",
])
def test_mash_dist_parser_valid(args):
    assert check_mash_dist_arguments(args) is not None


def test_check_booleans():
    args = check_foldseek_arguments()
    assert args.diag_score  # Example for default positive value
    assert not args.exhaustive_search  # Example for default negative value
