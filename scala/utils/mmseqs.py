import os


def mmseqs_clustering(fasta_file, output_path=None, seq_id_threshold=0.0):
    """
    Run MMSeqs2 on the input fasta file.

    Args:
        fasta_file (str): filepath of the file containing all the protein sequences to be clustered
        output_path (str): directory to store the results in
        seq_id_threshold (float): threshold to be applied to consider two sequences to be in the same cluster

    Returns:

    """
    # can we somehow individualize this based on the file ? --
    # see mmseqs documentation for details on additional options [ '-c', '1.0',]
    os.system(
        f"mmseqs "
        f"easy-linclust "
        f"{fasta_file} "
        f"{output_path} "
        f"{os.path.join(output_path, 'tmp')} "
        f"--cluster-mode 2 "
        f"--similarity-type 2 "
        f"--min-seq-id {seq_id_threshold} "
    )

    cluster_file = f'{output_path}_cluster.tsv'
    rep_seq_file = f'{output_path}_rep_seq.fasta'
    all_seq_file = f'{output_path}_all_seqs.fasta'

    return cluster_file, rep_seq_file, all_seq_file
