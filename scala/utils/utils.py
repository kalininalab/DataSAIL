import string
import random


def random_string(string_length=10):
    """
    Generate a random string of fixed length.

    Args:
        string_length (int): Number of characters in string

    Returns:
        random string of length string_length of lower case ascii characters
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(string_length))


def seq_map_to_fasta(seq_map, outfile, subset=None):
    """
    Write a set of sequences to a FASTA file.

    Args:
        seq_map (Dict[str, str]): Dictionary mapping sequences IDs to amino acid sequences
        outfile (str): Filepath where to write the sequences to
        subset (Iterable[str]): Iterable object holding the sequence identifies to be written to the FASTA file
    """
    if subset is None:
        iterator = seq_map
    else:
        iterator = subset

    with open(outfile, 'w') as f:
        for prot_id in iterator:
            f.write(f">{prot_id}")
            f.write(f"{seq_map[prot_id]}")


def parse_fasta(path=None, left_split=None, right_split=' ', check_dups=False):
    """
    Parse a FASTA file and do some validity checks if requested.

    Args:
        path:
        left_split:
        right_split:
        check_dups:

    Returns:
        Dictionary mapping sequences IDs to amino acid sequences
    """
    seq_map = {}

    with open(path, "r") as fasta:
        for line in fasta.readlines():
            if len(line) == 0:
                continue
            if line[0] == '>':
                entry_id = line[1:].replace('β', 'beta')

                if entry_id[:3] == 'sp|' or entry_id[:3] == 'tr|':  # Detect uniprot/tremble ID strings
                    entry_id = entry_id.split('|')[1]

                if left_split is not None:
                    entry_id = entry_id.split(left_split, 1)[1]
                if right_split is not None:
                    entry_id = entry_id.split(right_split, 1)[0]
                if check_dups and entry_id in seq_map:
                    print(f'Duplicate entry in fasta input detected: {entry_id}')
                seq_map[entry_id] = ''
            else:
                seq_map[entry_id] += line

    return seq_map


def get_cov_seq_ident(full_length, target_seq, template_seq):
    target_length = len(target_seq.replace("-", ""))
    template_length = target_length - template_seq.count("-")

    if template_length == 0:
        return None, None

    aln_length = template_length / full_length
    i = 0
    identical = 0

    for res_a in target_seq:
        if template_seq[i] == res_a:
            identical += 1
        i += 1

    seq_id = identical / template_length

    return aln_length, seq_id
