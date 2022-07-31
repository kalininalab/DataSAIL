import random
import os
import subprocess
import string

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def seqMapToFasta(seq_map, outfile, subset = None):
    lines = []

    if subset is None:
        iterator = seq_map
    else:
        iterator = subset

    for prot_id in iterator:

        seq = seq_map[prot_id]
        lines.append(f'>{prot_id}\n')
        lines.append(f'{seq}\n')

    if len(lines) > 0:
        f = open(outfile, 'w')
        f.write(''.join(lines))
        f.close()
        return True
    else:
        return 'Empty fasta file'

def parseFasta(path=None, new_file=None, lines=None, page=None, left_split=None, right_split=' '):
    if lines is None and page is None:
        f = open(path, 'r')
        lines = f.read().split('\n')
        f.close()
    elif lines is None:
        lines = page.split('\n')

    seq_map = {}
    n = 0

    if new_file is not None:
        new_lines = []

    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '>':
            entry_id = line[1:]
            if left_split is not None:
                entry_id = entry_id.split(left_split, 1)[1]
            if right_split is not None:
                entry_id = entry_id.split(right_split, 1)[0]
            seq_map[entry_id] = ''
            n += 1
            if new_file is not None:
                new_lines.append(line)
        else:
            seq_map[entry_id] += line
            if new_file is not None:
                new_lines.append(line)

    if new_file is not None:
        f = open(new_file, 'w')
        f.write('\n'.join(new_lines))
        f.close()

    return seq_map

def call_mmseqs_clustering(env, fasta_file, output_path = None, seq_id_threshold = 0.0, silenced = True):

    if output_path is None:
        infile_trunk = fasta_file.split('/')[-1].rsplit('.',1)[0]
        output_path = f'{env.out_dir}/{infile_trunk}'

    cmds = [env.mmseqs2_path, 'easy-linclust', fasta_file, output_path, env.tmp_folder, '--similarity-type', '2', '--cov-mode', '0', '-c', '1.0', '--min-seq-id', str(seq_id_threshold)]

    if silenced:
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(cmds, stdout=FNULL)
    else:
        p = subprocess.Popen(cmds)
    p.wait()

    cluster_file = f'{output_path}_cluster.tsv'
    rep_seq_file = f'{output_path}_rep_seq.fasta'
    all_seq_file = f'{output_path}_all_seqs.fasta'
    return cluster_file, rep_seq_file, all_seq_file


def check_input_file(env):
    ids = []
    dups = []
    record = []

    for seq in SeqIO.parse(env.input_file, "fasta"):
        ids.append(seq.id)
        record.append(seq)

    for id in ids:
        if ids.count(id) > 1:
            dups.append(id)

    if dups:
        alt_path = f"{env.tmp_folder}_alt_fasta.fasta"
        SeqIO.write(record, alt_path, "fasta")

    return dups
