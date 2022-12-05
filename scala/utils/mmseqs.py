import os
import subprocess

def call_mmseqs_clustering(env, fasta_file, output_path = None, seq_id_threshold = 0.0, silenced = True):

    if output_path is None:
        infile_trunk = fasta_file.split('/')[-1].rsplit('.',1)[0]
        output_path = f'{env.out_dir}/{infile_trunk}'

    cmds = [env.mmseqs2_path, 'easy-linclust', fasta_file, output_path, env.tmp_folder, '--similarity-type', '2', '--cov-mode', '0', '-c', '1.0', '--min-seq-id', str(seq_id_threshold)]

    if env.verbosity <= 2:
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(cmds, stdout=FNULL)
    else:
        print(f'Calling MMSEQS2:\n{cmds}')
        p = subprocess.Popen(cmds)
    p.wait()

    cluster_file = f'{output_path}_cluster.tsv'
    rep_seq_file = f'{output_path}_rep_seq.fasta'
    all_seq_file = f'{output_path}_all_seqs.fasta'
    return cluster_file, rep_seq_file, all_seq_file


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
        f"--cov-mode 0 "
        f"-c 1.0 "
        f"--cluster-mode 2 "
        f"--similarity-type 2 "
        f"--min-seq-id {seq_id_threshold} "
    )

    cluster_file = f'{output_path}_cluster.tsv'
    rep_seq_file = f'{output_path}_rep_seq.fasta'
    all_seq_file = f'{output_path}_all_seqs.fasta'

    return cluster_file, rep_seq_file, all_seq_file


