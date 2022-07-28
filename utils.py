import random

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def seqMapToFasta(seq_map, outfile):
    lines = []

    for prot_id in seq_map:

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
