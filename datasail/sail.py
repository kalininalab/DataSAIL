import argparse
import logging
import os.path

from datasail.run import bqp_main

verb_map = {
    "C": logging.CRITICAL,
    "F": logging.FATAL,
    "E": logging.ERROR,
    "W": logging.WARNING,
    "I": logging.INFO,
    "D": logging.DEBUG,
}


def parse_args():
    parser = argparse.ArgumentParser(
        prog="DataSAIL - Data Splitting Against Information Leaking",
        description="Data SAIL is a tool proving you with sophisticated splits of any type of data to challenge your "
                    "AI model. DataSAIL is able to compute several different splits of data preventing information "
                    "from leaking from the training set into the validation or test sets.",
    )
    parser.add_argument(
        "-i",
        "--inter",
        type=str,
        default=None,
        dest="inter",
        help="Path to TSV file of protein-ligand interactions."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        help="Output directory to store the splits in.",
    )
    parser.add_argument(
        "--to-sec",
        default=10,
        dest="max_sec",
        type=int,
        help="Maximal time to spend optimizing the objective in seconds. This does not include preparatory work such "
             "as parsing data and clustering the input."
    )
    parser.add_argument(
        "--to-sol",
        default=1000,
        dest="max_sol",
        type=int,
        help="Maximal number of solutions to compute until end of search (in case no optimum was found)."
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        default="W",
        type=str,
        choices=["C", "F", "E", "W", "I", "D"],
        dest='verbosity',
        help="Verbosity level of the program",
    )
    split = parser.add_argument_group("Splitting Arguments")
    split.add_argument(
        "-t",
        "--technique",
        type=str,
        choices=["R", "ICP", "ICD", "IC", "CCP", "CCD", "CC"],
        default="R",
        dest="technique",
        help="Select the mode to split the data. R: random split, ICP: identity-based cold-protein split, "
             "ICD: identity-based cold-drug split, IC: identity-based cold-drug-target split, "
             "CCP: similarity-based cold-protein split, SCD: similarity-based cold-drug split, "
             "SC: similarity-based cold-drug-protein split."
    )
    split.add_argument(
        "-s",
        "--splits",
        default=[0.7, 0.2, 0.1],
        nargs="+",
        type=float,
        dest="splits",
        help="Sizes of the individual splits the program shall produce.",
    )
    split.add_argument(
        "--names",
        default=None,
        dest="names",
        nargs="+",
        type=str,
        help="Names of the splits in order of the -s argument."
    )
    split.add_argument(
        "--limit",
        default=0.05,
        type=float,
        dest="limit",
        help="Multiplicative factor by how much the limits of the splits can be exceeded.",
    )
    prot = parser.add_argument_group("Protein Input Arguments")
    prot.add_argument(
        "-p",
        "--protein",
        type=str,
        dest="protein_data",
        default=None,
        help="Protein input to the program. This can either be the filepath to a FASTA file or a directory containing "
             "PDB files.",
    )
    prot.add_argument(
        "--protein-weights",
        type=str,
        dest="protein_weights",
        default=None,
        help="Custom weights of the proteins. The file has to have TSV format where every line is of the form "
             "[prot_id >tab< weight]. The prot_id has to match a protein id from the protein input argument.",
    )
    prot.add_argument(
        "--protein-sim",
        type=str,
        dest="protein_sim",
        default=None,
        help="Provide the name of a method to determine similarity between proteins or to cluster them. This can "
             "either be >WLK<, >mmseqs<, or a filepath to a file storing the pairwise similarities in TSV.",
    )
    prot.add_argument(
        "--protein-dist",
        type=str,
        dest="protein_dist",
        default=None,
        help="Provide the name of a method to determine distance between proteins or to cluster them. This can be a "
             "filepath to a file storing the pairwise distances in TSV."
    )
    prot.add_argument(
        "--protein-max-sim",
        type=float,
        dest="protein_max_sim",
        default=1.0,
        help="Maximum similarity of two proteins in two split."
    )
    prot.add_argument(
        "--protein-max-dist",
        type=float,
        dest="protein_max_dist",
        default=1.0,
        help="Maximal distance of two proteins in the same split."
    )
    lig = parser.add_argument_group("Ligand Input Arguments")
    lig.add_argument(
        "-l",
        "--ligand",
        type=str,
        dest="ligand_data",
        default=None,
        help="Ligand input to the program. This has to be a TSV file with every line as [lig_id >tab< SMILES]",
    )
    lig.add_argument(
        "--ligand-weights",
        type=str,
        dest="ligand_weights",
        default=None,
        help="Custom weights of the ligand. The file has to have TSV format where every line is of the form "
             "[lig_id >tab< weight]. The lig_id has to match a ligand id from the ligand input argument.",
    )
    lig.add_argument(
        "--ligand-sim",
        type=str,
        dest="ligand_sim",
        default=None,
        help="Provide the name of a method to determine similarity between ligands or to cluster them. This can "
             "either be >WLK< or a filepath to a file storing the pairwise similarities in TSV.",
    )
    lig.add_argument(
        "--ligand-dist",
        type=str,
        dest="ligand_dist",
        default=None,
        help="Provide the name of a method to determine distance between ligands. This has to be a filepath to a file "
             "storing the pairwise distances in TSV."
    )
    lig.add_argument(
        "--ligand-max-sim",
        type=float,
        dest="ligand_max_sim",
        default=1.0,
        help="Maximum similarity of two ligands in two split."
    )
    lig.add_argument(
        "--ligand-max-dist",
        type=float,
        dest="ligand_max_dist",
        default=1.0,
        help="Maximal distance of two ligands in the same split."
    )
    gene = parser.add_argument_group("Genomic Input Arguments")
    gene.add_argument(
        "-g",
        "--genomes",
        type=str,
        dest="genome_data",
        default=None,
        help="Genomic input to the program. This has to be a FASTA file.",
    )
    gene.add_argument(
        "--genome-weights",
        type=str,
        dest="genome_weights",
        default=None,
        help="Custom weights of the genomes. The file has to have TSV format where every line is of the form "
             "[gene_id >tab< weight]. The gene_id has to match a genome id from the genome input argument.",
    )
    gene.add_argument(
        "--genome-sim",
        type=str,
        dest="genome_sim",
        default=None,
        help="Provide the name of a method to determine similarity between genomes or to cluster them. This has to be "
             "a filepath to a file storing the pairwise similarities in TSV.",
    )
    gene.add_argument(
        "--genome-dist",
        type=str,
        dest="genome_dist",
        default=None,
        help="Provide the name of a method to determine distances between genomes. This can be >MASH< or a filepath to "
             "a file storing the pairwise distances in TSV."
    )
    gene.add_argument(
        "--genome-max-sim",
        type=float,
        dest="genome_max_sim",
        default=1.0,
        help="Maximum similarity of two genomes in two split."
    )
    gene.add_argument(
        "--genome-max-dist",
        type=float,
        dest="genome_max_dist",
        default=1.0,
        help="Maximal distance of two genomes in the same split."
    )
    other = parser.add_argument_group("Ligand Input Arguments")
    other.add_argument(
        "--other",
        type=str,
        dest="other_data",
        default=None,
        help="Non-standard input to the program. This is input that is neither proteins, chemical molecules, or "
             "genomic data. The provided argument has to be a TXT file with data IDs, one per line.",
    )
    other.add_argument(
        "--other-weights",
        type=str,
        dest="other_weights",
        default=None,
        help="Custom weights of the non-standard data. The file has to have TSV format where every line is of the form "
             "[id >tab< weight]. The id has to match an id from the --other input argument.",
    )
    other.add_argument(
        "--other-sim",
        type=str,
        dest="other_sim",
        default=None,
        help="Provide a filepath to a file storing the pairwise similarities between the non-standard data in TSV.",
    )
    other.add_argument(
        "--ligand-dist",
        type=str,
        dest="ligand_dist",
        default=None,
        help="Provide a filepath to a file storing the pairwise similarities between the non-standard data in TSV.",
    )
    other.add_argument(
        "--other-max-sim",
        type=float,
        dest="other_max_sim",
        default=1.0,
        help="Maximum similarity of two data points in two split."
    )
    other.add_argument(
        "--other-max-dist",
        type=float,
        dest="other_max_dist",
        default=1.0,
        help="Maximal distance of two data points in the same split."
    )
    return vars(parser.parse_args())


def error(msg, code):
    logging.error(msg)
    exit(code)


def validate_args(**kwargs):
    logging.basicConfig(level=verb_map[kwargs["verbosity"]])
    logging.info("Validating arguments")

    if not os.path.isdir(kwargs["output"]):
        logging.warning("Output directory does not exist, DataSAIL creates it automatically")
        os.makedirs(kwargs["output"], exist_ok=True)

    if len(kwargs["splits"]) < 2:
        error("Less then two splits required. This is no useful input, please check the input again.", 1)
    if kwargs["names"] is None:
        kwargs["names"] = [f"Split{x:03s}" for x in range(len(kwargs["splits"]))]
    elif len(kwargs["names"]) != len(kwargs["names"]):
        error("Different number of splits and names. You have to give the same number of splits and names for them.", 2)
    kwargs["splits"] = [x/sum(kwargs["splits"]) for x in kwargs["splits"]]

    return kwargs


def sail(**kwargs):
    kwargs = validate_args(**kwargs)
    bqp_main(**kwargs)


if __name__ == '__main__':
    sail(**parse_args())
